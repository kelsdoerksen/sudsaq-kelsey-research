"""
Run training, calibration, and testing on UNet with CQR

Train two models, upper and lower bounds according to alpha
Predict upper and lower on test set, assess the coverage according to true y_vals
Predict upper and lower on cal set
Calculate qyhat
Update upper and lowerbounds with qyhat calibration
"""
from model import *
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch import optim
from losses import *
from utils import *


def cqr_training_loop(model,
                      data_loader,
                      loss_criterion,
                      optimizer,
                      grad_scaler,
                      epochs,
                      experiment,
                      device,
                      model_type,
                      save_dir):
    global_step = 0
    epoch_number = 0
    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        model.train()
        epoch_loss = 0

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()

            outputs = model(inputs)  # predict on input

            loss = loss_criterion(outputs, labels)  # Calculate loss

            grad_scaler.scale(
                loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

        experiment.log({
            '{}_train Pinball loss'.format(model_type): epoch_loss / len(data_loader),
            '{}_step'.format(model_type): global_step,
            '{}_epoch'.format(model_type): epoch,
            '{}_optimizer'.format(model_type): 'adam'
        })

    # Saving model at end of epoch with experiment name
    out_model = '{}/{}_{}_last_epoch.pth'.format(save_dir, experiment.name, model_type)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               out_model)

    return out_model


def cqr_testing_loop(in_model,
                     model_type,
                     target,
                     test_dataset,
                     loss_criterion,
                     wandb_experiment,
                     channels,
                     out_dir,
                     device):
    """
    Predict standard way
    """

    # Setting model to eval mode
    pred_model = models.UNet(n_channels=int(channels), n_classes=1)
    pred_model.load_state_dict(torch.load(in_model)['state_dict'])
    pred_model.eval()

    loss_score = 0
    # iterate over the test set
    preds = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict
            outputs = pred_model(inputs)

            # Append first to preserve image shape for future plotting
            gt.append(labels.detach().numpy())
            preds.append(outputs.detach().numpy())

            loss_score += loss_criterion(outputs, labels)

    print('test set loss is: {}'.format(loss_score / len(test_dataset)))

    wandb_experiment.log({
        'Test set Pinball Loss_{}'.format(model_type): loss_score / len(test_dataset),
    })

    if model_type in ['lower_test', 'upper_test']:
        for i in range(len(gt)):
            np.save('{}/{}channels_{}_groundtruth_{}.npy'.format(out_dir, channels, target, i), gt[i])
            np.save('{}/{}channels_{}_pred_{}_{}.npy'.format(out_dir, channels, target, i, model_type), preds[i])

    return preds


def calculate_coverage(lower, upper, y_data):
    """
    Calculate coverage of prediction bound with true labels
    """
    y_true = []
    y_mask = []
    for i, data in enumerate(y_data):
        input, label = data
        # Mask nans, we don't care about these
        mask = ~torch.isnan(label)
        y_mask.append(mask)
        label = label[mask]
        label_np = label.numpy()
        label_list = list(label_np.flatten())
        y_true.extend(label_list)

    lower_list = []
    upper_list = []
    for i in range(len(lower)):
        low = lower[i]
        low = low[y_mask[i]]
        low_list = list(low.flatten())
        lower_list.extend(low_list)

        up = upper[i]
        up = up[y_mask[i]]
        up_list = list(up.flatten())
        upper_list.extend(up_list)

    out_of_bound = 0
    N = len(y_true)

    for i in range(N):
        if y_true[i] < lower_list[i] or y_true[i] > upper_list[i]:
            out_of_bound += 1

    return 1 - out_of_bound / N


def calibrate_qyhat(y_data, lower, upper, alpha):
    """
    Calibrate bounds with qyhat based on calibration predictions
    """
    y_true = []
    y_mask = []
    for i, data in enumerate(y_data):
        input, label = data
        # Mask nans, we don't care about these
        mask = ~torch.isnan(label)
        y_mask.append(mask)
        label = label[mask]
        label_np = label.numpy()
        label_list = list(label_np.flatten())
        y_true.extend(label_list)

    lower_list = []
    upper_list = []
    for i in range(len(lower)):
        low = lower[i]
        low = low[y_mask[i]]
        low_list = list(low.flatten())
        lower_list.extend(low_list)

        up = upper[i]
        up = up[y_mask[i]]
        up_list = list(up.flatten())
        upper_list.extend(up_list)

    N = len(y_true)
    s = np.amax([np.array(lower_list) - np.array(y_true), np.array(y_true) - np.array(upper_list)], axis=0)
    q_yhat = np.quantile(s, np.ceil((N + 1) * (1 - alpha)) / N)

    return q_yhat


def run_cqr(model,
            device,
            train_dataset,
            test_datatset,
            save_dir,
            experiment,
            alpha,
            channels,
            epochs: int,
            batch_size: int,
            learning_rate: float,
            weight_decay: float=0,
            save_checkpoint: bool=True):
    """
    Run CQR
    """
    # --- Split training dataset into training, and calibration
    n_cal = int(len(train_dataset) * 0.5)
    n_train = len(train_dataset) - n_cal
    train_set, cal_set = random_split(train_dataset, [n_train, n_cal], generator=torch.Generator().manual_seed(0))

    # --- DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    cal_loader = DataLoader(cal_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_datatset, batch_size=batch_size, shuffle=True)

    # --- Setting up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Setting up loss
    lower_criterion = NoNaNPinballLoss(alpha / 2)
    upper_criterion = NoNaNPinballLoss(1 - alpha / 2)

    # --- Setting up schedulers
    grad_scaler = torch.cuda.amp.GradScaler()

    # --- Fit lower and upper bound models on training data
    print('Training lower bound model...')
    lower_bound = cqr_training_loop(model, train_loader, lower_criterion, optimizer, grad_scaler, epochs, experiment,
                                    device, 'lower', save_dir)
    print('Training upper bound model...')
    upper_bound = cqr_training_loop(model, train_loader, upper_criterion, optimizer, grad_scaler, epochs, experiment,
                                    device, 'upper', save_dir)

    # --- Predict lower and upper on test set
    print('Predicting lower bounds...')
    uncal_lower_preds_t = cqr_testing_loop(lower_bound, 'lower_test', 'bias', test_loader, lower_criterion,
                     experiment, channels, save_dir, device)
    print('Predicting upper bounds...')
    uncal_upper_preds_t = cqr_testing_loop(lower_bound, 'upper_test', 'bias', test_loader, upper_criterion,
                     experiment, channels, save_dir, device)

    # --- Calculate the coverage from the un-calibrated predictions
    uncal_coverage = calculate_coverage(uncal_lower_preds_t, uncal_upper_preds_t, test_loader)
    print('Quantile Regression Coverage without conformal is: {}'.format(uncal_coverage))

    # --- Predict lower and upper on calibration set
    print('Predicting lower bounds on calibration set...')
    uncal_lower_preds_c = cqr_testing_loop(lower_bound, 'lower_cal', 'bias', cal_loader, lower_criterion,
                     experiment, channels, save_dir, device)
    print('Predicting upper bounds on calibration set...')
    uncal_upper_preds_c = cqr_testing_loop(upper_bound, 'upper_cal', 'bias', cal_loader, lower_criterion,
                     experiment, channels, save_dir, device)

    # --- Calculate qyhat
    print('Calculating qyhat for calibration...')
    qyhat = calibrate_qyhat(cal_loader, uncal_lower_preds_c, uncal_upper_preds_c, alpha)

    # --- Calibrate prediction intervals on the test set predictions
    print('Calibrating Test set predictions...')
    cal_lower_preds_t = []
    cal_upper_preds_t = []
    for i in range(len(uncal_lower_preds_t)):
        cal_lower_preds_t.append(uncal_lower_preds_t[i] - qyhat)
        cal_upper_preds_t.append(uncal_upper_preds_t[i] + qyhat)
        np.save('{}/{}channels_bias_pred_lower_cal_{}.npy'.format(save_dir, channels, i), cal_lower_preds_t[i])
        np.save('{}/{}channels_bias_pred_upper_cal_{}.npy'.format(save_dir, channels, i), cal_upper_preds_t[i])

    # --- Calculate coverage with calibrated predictions
    cal_coverage = calculate_coverage(cal_lower_preds_t, cal_upper_preds_t, test_loader)
    print('Conformalized Quantile Regression Coverage is: {}'.format(cal_coverage))

    experiment.log({
        'CQR coverage': cal_coverage
    })
