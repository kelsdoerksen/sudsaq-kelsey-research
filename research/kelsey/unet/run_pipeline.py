"""
Pipeline script to:
1. Train and validate model via train.py
2. Run predictions on test set via predict.py
3. Plot results using analysis/plotting_results.py

"""

import argparse
from predict import *
import logging
import sys
from run_cqr import *
from run_deep_ensemble import *
import random

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/cqr')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/nonconformist')))


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on AQ Dataset')
    parser.add_argument('--epochs', '-e',  type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Model optimizer')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--test_year', '-t', type=str, help='Test year for analysis (sets out of training)')
    parser.add_argument('--overfitting_test', type=str, help='Dir for overfitting model to check')
    parser.add_argument('--channels', required=True,
                        help='Number of channels, must be one of 9, 32, 39 for now')
    parser.add_argument('--target', help='target for unet, must be one of mda8 or bias',
                        required=True)
    parser.add_argument('--region', help='Region of study, NorthAmerica or Europe supported',
                        required=True)
    parser.add_argument('--model_type', help='Model type, must be one of standard, mcdropout, cqr, ensemble',
                        required=True)
    parser.add_argument('--data_dir', help='Specify root data directory',
                        required=True)
    parser.add_argument('--save_dir', help='Specify root save directory',
                        required=True),
    parser.add_argument('--val_percent', help='Validation percentage',
                        required=True)
    parser.add_argument('--norm', help='Normalization type for data preprocessing',)
    parser.add_argument('--tag', help='Wandb tag')
    parser.add_argument('--wandb_status', default='offline', help='Specify if offline or online experiment')
    parser.add_argument('--analysis_month', help='Month of analysis. Must be one of June, July, August.')
    parser.add_argument('--sensitivity_feature', help='Specify for feature to run sensitivity analysis on',
                        default=None)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    channels = int(args.channels)
    region = args.region
    target = args.target
    model_type = args.model_type
    root_data_dir = args.data_dir
    root_save_dir = args.save_dir
    test_year = args.test_year
    epochs = args.epochs
    tag = args.tag
    wandb_status = args.wandb_status
    analysis_month= args.analysis_month
    sensitivity_feature = args.sensitivity_feature

    # Want to be probabilistic in general
    seed = random.randint(0, 1000)
    torch.manual_seed(seed)

    # Initializing logging in wandb for experiment
    experiment = wandb.init(project='U-Net Test', resume='allow', anonymous='must', tags=[tag])

    # Check if running sensitivity analysis, if yes, remove one channel from count
    if sensitivity_feature:
        channel_count = channels-1
        experiment.config.update(
            dict(experiment='sensitivity_analysis',
                 sensitivity_feature=sensitivity_feature))
    else:
        channel_count = channels

    # Updating wandb experiment accordingly
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             val_percent=0.1, save_checkpoint=True,
             n_channels=channel_count, analysis_month=analysis_month,
             target=target, region=region, model=model_type, test_year=test_year, normalization=args.norm)
    )

    # --- Setting Directories
    sample_dir_root = '{}/{}/{}_channels'.format(root_data_dir, region, channels)
    label_dir_root = '{}/{}/labels_{}'.format(root_data_dir, region, target)

    # --- Making save directory
    if wandb_status == 'online':
        #save_dir = '{}/{}/{}/{}/{}_channels/{}'.format(root_save_dir, region, target, analysis_month, channels, experiment.name)
        save_dir =  '{}/{}'.format(root_save_dir, experiment.name)
    else:
        #save_dir = '{}/{}/{}/{}/{}_channels'.format(root_save_dir, region, target, analysis_month, channels)
        save_dir = '{}/{}'.format(root_save_dir, experiment.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if model_type in ['standard']:
        unet = models.UNet(n_channels=channel_count, n_classes=1)
        unet.to(device)
    elif model_type in ['cqr']:
        unet = models.CQRUNet(n_channels=channel_count, quantiles=[0.1, 0.5, 0.9])
        unet.to(device)
    elif model_type in ['mcdropout']:
        unet = models.MCDropoutProbabilisticUNet(n_channels=channel_count, n_classes=1)
        unet.to(device)

    # ---- Grabbing Data ----
    print('Grabbing training data...')
    aq_train_dataset = AQDataset(sample_dir_root, label_dir_root, 'train', analysis_month, test_year, channels, region,
                                 None, mean=None, std=None, max_val=None, min_val=None, norm=None)

    # Applying normalization
    print('Applying normalization...')
    n_channels = channels
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    global_min = torch.full((n_channels,), float('inf'))
    global_max = torch.full((n_channels,), float('-inf'))
    n_samples = 0

    if args.norm == 'zscore':
        print("Computing per-channel mean and std...")
        for images, _ in aq_train_dataset:
            images = images.float().contiguous()
            # Reshape to (32, 6000)
            images_flat = images.view(n_channels, -1)

            # Calculate mean and std
            mean += images_flat.mean(dim=1)
            std += images_flat.std(dim=1)
            n_samples += 1

        mean /= n_samples
        std /= n_samples
        max_val = None
        min_val = None

    if args.norm == 'minmax':
        print("Computing per-channel min and max...")
        for images, _ in aq_train_dataset:
            images = images.float().contiguous()
            # Reshape to (32, 6000)
            images_flat = images.view(n_channels, -1)

            # Calculating min and max
            min_vals = images_flat.min(dim=1).values
            max_vals = images_flat.max(dim=1).values

            global_min = torch.min(global_min, min_vals)
            global_max = torch.max(global_max, max_vals)
            mean = None
            std = None

    aq_train_dataset = AQDataset(sample_dir_root, label_dir_root, 'train', analysis_month, test_year, channels,
                                 region, sensitivity_feature, mean=mean, std=std, max_val= max_val, min_val= min_val,
                                 norm=args.norm)

    # --- Grabbing Testing Data ----
    print('Grabbing testing data...')
    aq_test_dataset = AQDataset(sample_dir_root, label_dir_root, 'test', analysis_month, test_year, channels,
                                 region, sensitivity_feature, mean=mean, std=std, max_val=max_val, min_val=min_val,
                                 norm=args.norm)

    if model_type == 'cqr':
        run_cqr(unet, device, aq_train_dataset, aq_test_dataset, save_dir, experiment, 0.1, channel_count,
                args.epochs, args.batch_size, args.lr, args.val_percent, 0, save_checkpoint=True)

    if model_type == 'ensemble':
        ensemble_size = 10
        run_deep_ensemble(device, aq_train_dataset, aq_test_dataset, 0.1, save_dir, channel_count,
                args.epochs, args.batch_size, args.lr, 0, ensemble_size=10)

    if model_type == 'standard':
        print('Training model...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model = train_model(
            model=unet,
            device=device,
            dataset=aq_train_dataset,
            save_dir=save_dir,
            experiment=experiment,
            val_percent=float(args.val_percent),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            opt = args.optimizer,
            save_checkpoint=True)

        print('Running Test set...')
        predict(trained_model, target, aq_test_dataset, experiment, channel_count, save_dir, device=device)

    if model_type == 'mcdropout':
        print('Training model...')
        trained_model = train_probabilistic_model(model=unet,
                                                  device=device,
                                                  dataset=aq_train_dataset,
                                                  save_dir=save_dir,
                                                  experiment=experiment,
                                                  epochs=args.epochs,
                                                  val_percent=float(args.val_percent),
                                                  batch_size=args.batch_size,
                                                  learning_rate=args.lr,
                                                  opt = args.optimizer,
                                                  save_checkpoint=True)

        print('Running Test set...')
        # Running probabilistic method to quanitfy UQ
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predict_probabilistic(trained_model, target, aq_test_dataset, experiment, channel_count, save_dir,
                              device=device)