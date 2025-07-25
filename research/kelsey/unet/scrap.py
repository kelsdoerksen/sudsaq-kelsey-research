# Model architectures
#
# Steven Lu
# August 2, 2021
import os
import sys
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models import D1UNet
from models import D1DUNet
from models import StandardUNet
from models import SDUNet
from models import MultiScaleUNet
from models import ConvNet
from models import WideUNet
from models import WideMultiScaleUNet

# Temporary solution to fix running the following code on Mac OS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

GVSModel = {
    'd1_unet': D1UNet,
    'd1d_unet': D1DUNet,
    'standard_unet': StandardUNet,
    'sd_unet': SDUNet,
    'multiscale_unet': MultiScaleUNet,
    'conv_net': ConvNet,
    'wide_unet': WideUNet,
    'wide_multiscale_unet': WideMultiScaleUNet
}

def make_deterministic(seed):
    # Making Pytorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
​
    # Making Python deterministic
    random.seed(seed)
​
    # Making numpy deterministic
    np.random.seed(seed)
​
​
def get_runtime_device(device, device_id=None):
    if device.lower() == 'cpu':
        runtime_device = 'cpu'
    elif device.lower() == 'gpu':
        if torch.cuda.is_available():
            if device_id is None:
                runtime_device = 'cuda'
            else:
                runtime_device = 'cuda:%s' % device_id
        else:
            print('[ERROR] GPU is not available on this machine. Set device to '
                  'cpu.')
            sys.exit(1)
    else:
        print('[ERROR] Unrecognized device. Device can only be GPU or CPU.')
        sys.exit(1)
​
    return runtime_device
​
​
class GvsDataset(Dataset):
    def __init__(self, root_dir, tile_file, mean_arr, std_arr):
        with open(tile_file, 'r') as f:
            tile_list = f.readlines()
            tile_list = [t.strip() for t in tile_list]
        self.tile_files = [os.path.join(root_dir, f) for f in tile_list]
        self.mean_arr = mean_arr
        self.std_arr = std_arr
​
    def __len__(self):
        return len(self.tile_files)
​
    def __getitem__(self, index):
        path_id = self.tile_files[index]
        file_id = os.path.basename(path_id)
        layers = np.load('%s_layer.npy' % path_id)
        layers = (layers - self.mean_arr) / self.std_arr
​
        # The sizes of the tiles we generated are in the format of (h, w, c),
        # whereas pytorch expects the sizes to be in the format of (c, h, w).
        layers = np.moveaxis(layers, 2, 0)
​
        lidar_mch = np.array([np.load('%s_lidar.npy' % self.tile_files[index])])
​
        return layers.astype(np.float32), lidar_mch.astype(np.float32), file_id
​
​
def log_mse_loss(preds, truth, w):
    sum = torch.sum((preds - truth)**2)
    mean = sum / torch.numel(preds)
    log_sum = torch.sum((torch.log(preds + 1e-9) - torch.log(truth + 1e-9))**2)
    log_mean = log_sum / torch.numel(preds)
    loss = mean + w * log_mean
​
    return loss
​
​
def main(model_architecture, root_dir, train_list, val_list, mean_file, std_file,
         log_file, out_dir, n_epoch, batch_size, device, device_id, seed):
    # make_deterministic(seed)
    runtime_device = get_runtime_device(device, device_id)
​
    fmt = logging.Formatter(fmt='%(asctime)-15s: %(message)s',
                            datefmt='[%Y-%m-%d %H:%M:%S]')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(fmt)
    logger = logging.getLogger('GVS')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
​
    mean_arr = np.load(mean_file)
    std_arr = np.load(std_file)
​
    train_dataset = GvsDataset(root_dir, train_list, mean_arr, std_arr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
​
    val_dataset = GvsDataset(root_dir, val_list, mean_arr, std_arr)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
​
    model = GVSModel[model_architecture]()
    model.to(runtime_device)
    print('Model parameters: %d' % sum(p.numel() for p in model.parameters()
                                       if p.requires_grad))
​
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.L1Loss(reduction='mean')
    # criterion = torch.nn.MSELoss(reduction='mean')
​
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
​
    for epoch in tqdm(range(n_epoch), desc='Epoch'):
        model.train()
        train_loss = 0.0
        for ind, (layers, lidars, _) in enumerate(
                tqdm(train_loader, leave=False, desc='Iteration',
                     total=len(train_dataset) // batch_size)):
​
            layers = layers.to(runtime_device)
            lidars = lidars.to(runtime_device)
            mask = ~torch.isnan(lidars)
            with torch.set_grad_enabled(True):
                preds = model(layers)

                preds = preds[mask]
                lidars = lidars[mask]

                loss = criterion(preds, lidars)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            logger.info('[Iter: %-10d Epoch:%-3d] train loss = %f' %
                        (ind, epoch, loss.item()))

        model.eval()
        val_loss = 0.0
        val_iter = 0
        for val_layers, val_lidars, _ in tqdm(val_loader, leave=False,
                                              desc='Eval on validation set'):
            val_iter += 1
            val_layers = val_layers.to(runtime_device)
            val_lidars = val_lidars.to(runtime_device)
            val_mask = ~torch.isnan(val_lidars)

            with torch.set_grad_enabled(False):
                val_preds = model(val_layers)

                val_preds = val_preds[val_mask]
                val_lidars = val_lidars[val_mask]

                v_loss = criterion(val_preds, val_lidars)

            val_loss += v_loss.item()
        scheduler.step(val_loss / val_iter)
        logger.info('[Iter: %-10d Epoch:%-3d] validation loss = %f' %
                    (ind, epoch, val_loss / val_iter))

        out_model = '%s/gvs_epoch_%d.pth' % (out_dir, epoch)
        torch.save(model, out_model)
        logger.info('Saved model: %s' % os.path.abspath(out_model))

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #         print(param.data)

    # out_model = '%s/gvs.pth' % out_dir
    # torch.save(model, out_model)
    # logger.info('Saved model: %s' % os.path.abspath(out_model))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_architecture',
                        choices=['d1d_unet', 'd1_unet', 'standard_unet',
                                 'multiscale_unet', 'sd_unet', 'conv_net',
                                 'wide_unet', 'wide_multiscale_unet'],
                        type=str)
    parser.add_argument('root_dir', type=str)
    parser.add_argument('train_list', type=str)
    parser.add_argument('val_list', type=str)
    parser.add_argument('mean_file', type=str)
    parser.add_argument('std_file', type=str)
    parser.add_argument('log_file', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--n_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--device', default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--device_id', default='0', choices=['0', '1', '2', '3'])
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    main(**vars(args))