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
import random


def cqr_training_loop(model,
                      data_loader,
                      val_data_loader,
                      loss_criterion,
                      optimizer,
                      grad_scaler,
                      epochs,
                      experiment,
                      device,
                      model_type,
                      save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

            outputs = model(inputs)  # predict quantiles on input
            loss = loss_criterion(outputs, labels)  # Calculate loss

            grad_scaler.scale(
                loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

        experiment.log({
            '{}_train Quantile loss'.format(model_type): epoch_loss / len(data_loader),
            '{}_step'.format(model_type): global_step,
            '{}_epoch'.format(model_type): epoch,
            '{}_optimizer'.format(model_type): 'adam'
        })

        print('{}_train Quantile loss is: {}'.format(model_type, epoch_loss / len(data_loader)))

        # Run validation
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            for k, vdata in enumerate(val_data_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_data_loader)
        # scheduler.step(avg_vloss)

        try:
            experiment.log({
                '{}_val Quantile loss'.format(model_type): avg_vloss
            })
        except:
            pass

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
                     alpha,
                     test_dataset,
                     loss_criterion,
                     wandb_experiment,
                     channels,
                     out_dir,
                     device):
    """
    Predict standard way
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setting model to eval mode
    pred_model = models.CQRUNet(n_channels=int(channels), quantiles = [float(alpha/2), 0.5, float(1-(alpha/2))])
    pred_model.load_state_dict(torch.load(in_model)['state_dict'])
    pred_model.to(device)
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
            gt.append(labels.cpu().detach().numpy())
            preds.append(outputs.cpu().detach().numpy())

            loss_score += loss_criterion(outputs, labels)

    print('test set loss is: {}'.format(loss_score / len(test_dataset)))

    wandb_experiment.log({
        'Test set Quantile Loss_{}'.format(model_type): loss_score / len(test_dataset),
    })

    for i in range(len(gt)):
        np.save('{}/{}channels_{}_pred_{}_{}_lower.npy'.format(out_dir, channels, target, model_type, i),
                preds[i][:,0,:,:])
        np.save('{}/{}channels_{}_pred_{}_{}_med.npy'.format(out_dir, channels, target, model_type, i),
                preds[i][:, 1, :, :])
        np.save('{}/{}channels_{}_pred_{}_{}_upper.npy'.format(out_dir, channels, target, model_type, i),
                preds[i][:, 2, :, :])

    return preds


def calculate_coverage(predictions, y_data, calibration_status):
    """
    Calculate coverage of prediction bound with true labels
    :param: predictions: model predictions
    :param: y_data: groundtruth data
    :param: calibration_status: uncalibrated or calibrated
    predictions to calculate coverage for
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
    if calibration_status == 'uncalibrated':
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                low = predictions[i][j,0,:,:]

                low = low[y_mask[i][j][0]]
                low_list = list(low.flatten())
                lower_list.extend(low_list)

                up = predictions[i][j,2,:,:]
                up = up[y_mask[i][j][0]]
                up_list = list(up.flatten())
                upper_list.extend(up_list)
    else:
        low_preds = predictions[0]
        upper_preds = predictions[1]
        for i in range(len(low_preds)):
            for j in range(len(low_preds[i])):
                low = low_preds[i][j,0,:,:]
                low = low[y_mask[i][j][0]]
                low_list = list(low.flatten())
                lower_list.extend(low_list)

                up = upper_preds[i][j, 2, :, :]
                up = up[y_mask[i][j][0]]
                up_list = list(up.flatten())
                upper_list.extend(up_list)

    out_of_bound = 0
    N = len(y_true)

    for i in range(N):
        if y_true[i] < lower_list[i] or y_true[i] > upper_list[i]:
            out_of_bound += 1

    return 1 - out_of_bound / N


def calibrate_qyhat(y_data, predictions, alpha):
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
    for i in range(len(predictions)):
        low = predictions[i][:,0,:,:]
        low = low[y_mask[i][:,0,:,:]]
        low_list = list(low.flatten())
        lower_list.extend(low_list)

        up = predictions[i][:,2,:,:]
        up = up[y_mask[i][:,0,:,:]]
        up_list = list(up.flatten())
        upper_list.extend(up_list)

    N = len(y_true)
    s = np.amax([np.array(lower_list) - np.array(y_true), np.array(y_true) - np.array(upper_list)], axis=0)
    q_yhat = np.quantile(s, np.ceil((N + 1) * (1 - alpha)) / N)

    return q_yhat


def get_mse(predictions, groundtruth):
    y_true = []
    y_mask = []
    for i, data in enumerate(groundtruth):
        input, label = data
        # Mask nans, we don't care about these
        mask = ~torch.isnan(label)
        y_mask.append(mask)
        label = label[mask]
        label_np = label.numpy()
        label_list = list(label_np.flatten())
        y_true.extend(label_list)

    preds_list = []
    for i in range(len(predictions)):
        pred = predictions[i][:, 0, :, :]
        pred = pred[y_mask[i][:, 0, :, :]]
        pred_list = list(pred.flatten())
        preds_list.extend(pred_list)

    return np.mean((np.array(preds_list) - np.array(y_true))**2)

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
            val_percent: float,
            weight_decay: float=0,
            save_checkpoint: bool=True):
    """
    Run CQR
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    # --- Split training dataset into training, and calibration
    n_cal = int(len(train_dataset) * 0.5)
    n_train = len(train_dataset) - n_cal
    train_set, cal_set = random_split(train_dataset, [n_train, n_cal], generator=torch.Generator().manual_seed(seed))

    # --- Split training dataset into train and val to monitor for overfitting
    if val_percent == 0:
        n_train = len(train_set)
        train_set = train_set
    else:
        n_val = int(len(train_set) * 0.1)
        n_train_final = len(train_set) - n_val
        train_set, val_set = random_split(train_set, [n_train_final, n_val], generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # --- DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    cal_loader = DataLoader(cal_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_datatset, batch_size=batch_size, shuffle=False)   # Set shuffle to false to preserve order of data for timeseries generation

    # --- Setting up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- Getting quantiles from alpha
    lower_q = alpha/2
    upper_q = 1-(alpha/2)

    quantiles = [lower_q, 0.5, upper_q]
    criterion = NoNaNQuantileLoss(quantiles)
    grad_scaler = torch.cuda.amp.GradScaler()
    trained_model = cqr_training_loop(model, train_loader, val_loader, criterion, optimizer, grad_scaler, epochs, experiment,
                                    device, 'quantiles', save_dir)

    # Predict on test set
    uncal_predictions_t = cqr_testing_loop(trained_model, 'uncalibrated', 'bias', alpha, test_loader, criterion, experiment,
                                   channels, save_dir, device)

    # --- Calculate the coverage from the un-calibrated predictions
    uncal_coverage = calculate_coverage(uncal_predictions_t, test_loader, 'uncalibrated')
    print('Quantile Regression Coverage without conformal is: {}'.format(uncal_coverage))

    # --- Predict lower and upper on calibration set
    print('Predicting on calibration set...')
    uncal_predictions_c = cqr_testing_loop(trained_model, 'calibration_samples', 'bias', alpha, cal_loader, criterion,
                                           experiment, channels, save_dir, device)

    # --- Calculate qyhat
    print('Calculating qyhat for calibration...')
    qyhat = calibrate_qyhat(cal_loader, uncal_predictions_c, alpha)

    # --- Calibrate prediction intervals on the test set predictions
    print('Calibrating Test set predictions...')
    cal_lower_preds_t = []
    cal_upper_preds_t = []
    cal_med_preds_t = []
    for i in range(len(uncal_predictions_t)):
        cal_lower_preds_t.append(uncal_predictions_t[i] - qyhat)
        cal_upper_preds_t.append(uncal_predictions_t[i] + qyhat)
        cal_med_preds_t.append(uncal_predictions_t[i])
        np.save('{}/{}channels_bias_pred_lower_cal_{}.npy'.format(save_dir, channels, i), cal_lower_preds_t[i])
        np.save('{}/{}channels_bias_pred_upper_cal_{}.npy'.format(save_dir, channels, i), cal_upper_preds_t[i])
        np.save('{}/{}channels_bias_pred_med_cal_{}.npy'.format(save_dir, channels, i), cal_med_preds_t[i])

    gt = []
    for i, data in enumerate(test_loader):
        inputs, labels = data
        # Append first to preserve image shape for future plotting
        gt.append(labels.cpu().detach().numpy())
    for i in range(len(gt)):
        np.save('{}/{}channels_bias_groundtruth_{}.npy'.format(save_dir, channels, i), gt[i])

    # --- Calculate coverage with calibrated predictions
    cal_coverage = calculate_coverage([cal_lower_preds_t, cal_upper_preds_t], test_loader, 'calibrated')
    print('Conformalized Quantile Regression Coverage is: {}'.format(cal_coverage))

    # --- Calculcate RMSE with calibrated predictions
    mse = get_mse(cal_med_preds_t, test_loader)
    rmse = np.sqrt(mse)
    print('RMSE is: {}'.format(rmse))

    experiment.log({
        'CQR coverage': cal_coverage,
        'RMSE': rmse
    })

    '''
    # Old, training individual models
    # Setting up loss
    lower_criterion = NoNaNPinballLoss(alpha / 2)
    upper_criterion = NoNaNPinballLoss(1 - alpha / 2)

    # --- Setting up schedulers
    grad_scaler = torch.cuda.amp.GradScaler()

    # --- Fit lower and upper bound models on training data
    print('Training lower bound model...')
    lower_bound = cqr_training_loop(model, train_loader, val_loader, lower_criterion, optimizer, grad_scaler, epochs, experiment,
                                    device, 'lower', save_dir)
    print('Training upper bound model...')
    upper_bound = cqr_training_loop(model, train_loader, val_loader, upper_criterion, optimizer, grad_scaler, epochs, experiment,
                                    device, 'upper', save_dir)

    # --- Predict lower and upper on test set
    print('Predicting lower bounds...')
    uncal_lower_preds_t = cqr_testing_loop(lower_bound, 'lower_test', 'bias', test_loader, lower_criterion,
                     experiment, channels, save_dir, device)
    print('Predicting upper bounds...')
    uncal_upper_preds_t = cqr_testing_loop(upper_bound, 'upper_test', 'bias', test_loader, upper_criterion,
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
    '''
