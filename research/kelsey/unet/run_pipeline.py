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
from torch.utils.data import ConcatDataset

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/cqr')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/nonconformist')))


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on AQ Dataset')
    parser.add_argument('--epochs', '-e',  type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Model optimizer')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--test-year', '-t', type=str, help='Test year for analysis (sets out of training)')
    parser.add_argument('--overfitting_test', type=str, help='Dir for overfitting model to check')
    parser.add_argument('--channels', required=True,
                        help='Number of channels, must be one of 9, 32, 39 for now')
    parser.add_argument('--target', help='target for unet, must be one of mda8 or bias',
                        required=True)
    parser.add_argument('--region', help='Region of study, NorthAmerica or Europe supported',
                        required=True)
    parser.add_argument('--seed', help='Seed to set to make model deterministic',
                        required=True)
    parser.add_argument('--model_type', help='Model type, must be one of standard, mcdropout, concrete, cqr',
                        required=True)
    parser.add_argument('--data_dir', help='Specify root data directory',
                        required=True)
    parser.add_argument('--save_dir', help='Specify root save directory',
                        required=True),
    parser.add_argument('--val_percent', help='Validation percentage',
                        required=True)
    parser.add_argument('--tag', help='Wandb tag')
    parser.add_argument('--wandb_status', default='offline', help='Specify if offline or online experiment')
    parser.add_argument('--analysis_date', help='Month of analysis')
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
    seed = args.seed
    analysis_time = args.analysis_date
    sensitivity_feature = args.sensitivity_feature

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
             n_channels=channel_count, analysis_period=analysis_time,
             target=target, region=region, model=model_type, test_year=test_year)
    )

    # --- Setting Directories
    sample_dir_root = '{}/{}/zscore_normalization'.format(root_data_dir, region)
    label_dir_root = '{}/{}/labels_{}'.format(root_data_dir, region, target)

    # --- Making save directory
    if wandb_status == 'online':
        save_dir = '{}/{}/{}/{}channels/{}_{}_{}_{}_{}epochs'.format(root_save_dir, region, target, channels, experiment.name,
                                                         region, analysis_time, model_type, epochs)
    else:
        save_dir = '{}/{}/{}/{}channels/{}_{}_{}_{}_{}epochs'.format(root_save_dir, region, target, channels,
                                                                     experiment.id,
                                                                     region, analysis_time, model_type, epochs)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if model_type in ['standard']:
        unet = models.UNet(n_channels=channel_count, n_classes=1)
    if model_type in ['cqr']:
        unet = models.CQRUNet(n_channels=channel_count, quantiles=[0.1, 0.5, 0.9])
    if model_type in ['mcdropout']:
        unet = models.MCDropoutProbabilisticUNet(n_channels=channel_count, n_classes=1)

    if torch.cuda.is_available():
        unet.cuda()
    #unet.to(device=device)

    # ---- Grabbing Training Data ----
    # Hardcoding to be test year either 2019 or 2020
    years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    train_years = years
    train_years.remove(int(test_year))
    if test_year == '2019':
        train_years.remove(int(test_year)+1)
    if test_year == '2016':
        train_years.remove(2020)
        train_years.remove(2019)
        train_years.remove(2018)
        train_years.remove(2017)
    print('Grabbing training data...')
    aq_train = []
    for y in train_years:
        train_sample_dir = '{}/{}_channels/{}/{}'.format(sample_dir_root, channels, analysis_time, y)
        train_label_dir = '{}/{}/{}'.format(label_dir_root, analysis_time, y)
        aq_dataset = AQDataset(train_sample_dir, train_label_dir, channels, region, sensitivity_feature)
        aq_train.append(aq_dataset)

    aq_train_dataset = ConcatDataset(aq_train)

    # --- Grabbing Testing Data ----
    print('Grabbing testing data...')
    root_sample_test_dir = '{}/{}/zscore_normalization'.format(root_data_dir, region)
    root_label_dir = '{}/{}/labels_{}'.format(root_data_dir, region, target)

    img_dir_test = '{}/{}_channels/{}/{}'.format(root_sample_test_dir, channels, analysis_time, test_year)
    label_dir_test = '{}/{}/{}'.format(root_label_dir, analysis_time, test_year)

    aq_test_dataset = AQDataset(img_dir_test, label_dir_test, channels, region, sensitivity_feature)

    if model_type == 'cqr':
        run_cqr(unet, device, aq_train_dataset, aq_test_dataset, save_dir, experiment, 0.1, channel_count,
                args.epochs, args.batch_size, args.lr, 0, save_checkpoint=True)


    if model_type == 'standard':
        print('Training model...')
        trained_model = train_model(
            model=unet,
            device=device,
            dataset=aq_train_dataset,
            save_dir=args.save_dir,
            experiment=experiment,
            val_percent=float(args.val_percent),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            opt = args.optimizer,
            save_checkpoint=True)

        print('Running Test set...')
        predict(trained_model, target, aq_test_dataset, experiment, channel_count, seed, save_dir, device=device)

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
        predict_probabilistic(trained_model, target, aq_test_dataset, experiment, channel_count, seed, save_dir,
                              device=device)