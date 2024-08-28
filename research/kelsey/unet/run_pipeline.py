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

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/cqr')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/nonconformist')))


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on AQ Dataset')
    parser.add_argument('--epochs', '-e',  type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Model optimizer')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--test-year', '-t', type=str, help='Test year for analysis (sets out of training)')
    parser.add_argument('--overfitting_test', type=str, help='Dir for overfitting model to check')
    parser.add_argument('--analysis_date', required=True,
                        help='Analysis date for model, must be one of june, july, august or summer')
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

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    channels = int(args.channels)
    analysis_time = args.analysis_date
    region = args.region
    target = args.target
    seed = int(args.seed)
    model_type = args.model_type
    root_data_dir = args.data_dir
    root_save_dir = args.save_dir
    test_year = args.test_year

    make_deterministic(seed)

    # Initializing logging in wandb for experiment
    experiment = wandb.init(project='U-Net Test', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             val_percent=0.1, save_checkpoint=True,
             n_channels=channels, analysis_period=analysis_time,
             target=target)
    )

    # --- Setting Directories
    sample_dir_root = '{}/{}/zscore_normalization'.format(root_data_dir, region)
    label_dir_root = '{}/{}/labels_{}'.format(root_data_dir, region, target)

    # --- Making save directory
    save_dir = '{}/{}/{}/{}channels/{}_{}_{}_{}'.format(root_save_dir, region, target, channels, experiment.name,
                                                     region, analysis_time, model_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if model_type in ['standard', 'cqr']:
        unet = models.UNet(n_channels=channels, n_classes=1)
    if model_type in ['mcdropout']:
        unet = models.MCDropoutProbabilisticUNet(n_channels=channels, n_classes=1)
    '''
    logging.info(f'Network:\n'
                 f'\t{unet.n_channels} input channels\n'
                 f'\t{unet.n_classes} output channels (classes)')
    '''

    if torch.cuda.is_available():
        unet.cuda()
    #unet.to(device=device)

    # ---- Grabbing Training Data ----
    # Training on 2005-2019, test on 2020
    print('Grabbing training data...')
    train_sample_dir = '{}/{}_channels/{}/2005-2019'.format(sample_dir_root, channels, analysis_time)
    train_label_dir = '{}/{}/2005-2019'.format(label_dir_root, analysis_time)

    aq_train_dataset = AQDataset(train_sample_dir, train_label_dir, channels, region)

    # --- Grabbing Testing Data ----
    # Hardcoded for 2020
    print('Grabbing testing data...')

    # Testing on 2020 Summer months (June, July August)
    root_sample_test_dir = '{}/{}/zscore_normalization'.format(root_data_dir, region)
    root_label_dir = '{}/{}/labels_{}'.format(root_data_dir, region, target)

    img_dir_test = '{}/{}_channels/{}/{}'.format(root_sample_test_dir, channels, analysis_time, test_year)
    label_dir_test = '{}/{}/{}'.format(root_label_dir, analysis_time, test_year)

    aq_test_dataset = AQDataset(img_dir_test, label_dir_test, channels, region)

    if model_type == 'cqr':
        run_cqr(unet, device, aq_train_dataset, aq_test_dataset, save_dir, experiment, 0.1, args.channels,
                args.epochs, args.batch_size, args.lr, 0, save_checkpoint=True)


    print('Training model...')
    if model_type == 'standard':
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
        predict(trained_model, target, aq_test_dataset, experiment, channels, seed, save_dir, device=device)

    if model_type == 'mcdropout':
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
        predict_probabilistic(trained_model, target, aq_test_dataset, experiment, channels, seed, save_dir,
                              device=device)