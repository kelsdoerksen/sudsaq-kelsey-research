"""
AQ dataset module
"""

from torch.utils.data import Dataset
from torchvision import transforms
from numpy import load, sort
import os
import torch
import numpy as np
from datetime import datetime, timedelta


def daterange(date1, date2):
    date_list = []
    date1_date = datetime.strptime(date1, "%Y-%m-%d")
    date2_date = datetime.strptime(date2, "%Y-%m-%d")
    for n in range(int((date2_date - date1_date).days)):
        dt = date1_date + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list

class AQDataset(Dataset):
    def __init__(self, image_dir, label_dir, split, month, test_year, channels, region, sensitivity, mean=None,
                 std=None, max_val=None, min_val=None, norm=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.channels = channels
        self.region = region
        self.mean = mean
        self.std = std
        self.max_val = max_val
        self.min_val = min_val
        self.norm = norm
        self.sensitivity = sensitivity
        self.split = split
        self.test_year = test_year
        self.month = month

        valid_years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]

        # Map the months to the value
        month_dict = {'June': 6, 'July': 7, 'August': 8}

        valid_date_list = []
        if split == 'train':
            train_dates = []
            train_years = [x for x in valid_years if x <= int(test_year)]
            for y in train_years:
                valid_date_list.extend(daterange('{}-0{}-01'.format(y, month_dict[self.month]), '{}-{}-01'.
                                             format(y, month_dict[self.month]+1)))
        if split == 'test':
            valid_date_list.extend(daterange('{}-{}-01'.format(test_year, month_dict[self.month]), '{}-{}-01'
                                             .format(test_year, month_dict[self.month]+1)))

        # Get list of dates from samples
        fns = os.listdir(image_dir)
        fns = [s.strip('_sample.npy') for s in fns]
        fns_valid = [x for x in fns if x in valid_date_list]

        self.image_fns = [x + '_sample.npy' for x in fns_valid]
        self.label_fns = [x + '_label.npy' for x in fns_valid]

    def __len__(self):
        if self.split == 'train':
            self.image_fns = [x for x in self.image_fns if "{}".format(self.test_year) not in x]
        elif self.split == 'test':
            self.image_fns = [x for x in self.image_fns if "{}".format(self.test_year) in x]
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fns = sort(self.image_fns)
        label_fns = sort(self.label_fns)

        if self.split == 'train':
            image_fns = [x for x in self.image_fns if "{}".format(self.test_year) not in x]
        elif self.split == 'test':
            image_fns = [x for x in self.image_fns if "{}".format(self.test_year) in x]

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

        # Apply normalization
        if self.mean is not None and self.std is not None:
            if self.norm == 'zscore':
                # Ensure shape (nchannels, 1, 1) for broadcasting
                mean = self.mean.view(-1, 1, 1)
                std = self.std.view(-1, 1, 1)
                multichannel_image = (multichannel_image - mean) / (std + 1e-8)
        if self.max_val is not None and self.min_val is not None:
            if self.norm == 'minmax':
                min_val = self.min_val.view(-1, 1, 1)
                max_val = self.max_val.view(-1, 1, 1)
                multichannel_image = (multichannel_image - min_val) / (max_val - min_val + 1e-8)

        # If applying sensitivity experiment, remove certain variables
        if self.sensitivity == 'momo.t':
            m1 = multichannel_image[0:19,:,:]
            m2 = multichannel_image[20:,:,:]
            multichannel_image = np.concatenate((m1, m2))
            self.channels = self.channels - 1
        if self.sensitivity == 'momo.ch20':
            m1 = multichannel_image[0:5, :, :]
            m2 = multichannel_image[6:, :, :]
            multichannel_image = np.concatenate((m1, m2))
            self.channels = self.channels - 1
        if self.sensitivity == 'momo.no2':
            m1 = multichannel_image[0:23, :, :]
            m2 = multichannel_image[24:, :, :]
            multichannel_image = np.concatenate((m1, m2))
            self.channels = self.channels - 1
        if self.sensitivity == 'momo.oh':
            m1 = multichannel_image[0:11, :, :]
            m2 = multichannel_image[12:, :, :]
            multichannel_image = np.concatenate((m1, m2))
            self.channels = self.channels - 1

        return multichannel_image.float(), label_class.float()

    def transform(self, image):
        transform_ops = transforms.ToTensor()
        return transform_ops(image)
