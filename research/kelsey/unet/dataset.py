"""
AQ dataset module
"""

from torch.utils.data import Dataset
from torchvision import transforms
from numpy import load, sort
import os
import torch

class AQDataset(Dataset):
    def __init__(self, image_dir, label_dir, channels, region):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.channels = channels
        self.region = region
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fns = sort(self.image_fns)
        label_fns = sort(self.label_fns)
        image_fn = image_fns[index]
        label_fn = label_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        label_fp = os.path.join(self.label_dir, label_fn)
        multichannel_image = load('{}'.format(image_fp), allow_pickle=True).astype('float32')
        label_class = load('{}'.format(label_fp), allow_pickle=True)
        multichannel_image = self.transform(multichannel_image)
        label_class = self.transform(label_class)

        if self.region == 'NorthAmerica':
            x = 31
            y = 49
        if self.region == 'Europe':
            x = 27
            y = 31
        if self.region == 'Globe':
            x = 160
            y = 320

        if multichannel_image.shape != torch.Size([self.channels, x, y]):
            multichannel_image = torch.transpose(multichannel_image, 0, 1)
            multichannel_image = torch.transpose(multichannel_image, 1, 2)
        if label_class.shape != torch.Size([1, x, y]):
            label_class = torch.transpose(label_class, 0, 1)
            label_class = torch.transpose(label_class, 1, 2)
        return multichannel_image.float(), label_class.float()

    def transform(self, image):
        transform_ops = transforms.ToTensor()
        return transform_ops(image)
