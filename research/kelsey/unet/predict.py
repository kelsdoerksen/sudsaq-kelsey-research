"""
Script to predict on test set after training model
"""

from dataset import *
import numpy as np
from model import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader, random_split
from torch import optim
from pathlib import Path
from train import *
from operator import add
from losses import *
from utils import *


def predict(in_model, target, test_dataset, wandb_experiment, channels, seed, out_dir, device):
    """
    Predict standard way (no dropout at test time)
    """
    # Make deterministic
    make_deterministic(seed)

    # Setting model to eval mode
    unet = models.UNet(n_channels=channels, n_classes=1)
    unet.load_state_dict(torch.load(in_model)['state_dict'])
    unet.eval()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    loss_criterion = nn.MSELoss()
    mse_score = 0
    # iterate over the test set
    preds = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict the mask
            outputs = unet(inputs)
            test_mask = ~torch.isnan(labels)

            # Append first to preserve image shape for future plotting
            gt.append(labels.detach().numpy())
            preds.append(outputs.detach().numpy())

            outputs = outputs[test_mask]
            labels = labels[test_mask]

            mse_score += loss_criterion(outputs, labels)

    print('test set mse is: {}'.format(mse_score / len(test_loader)))
    print('test set rmse is: {}'.format(np.sqrt((mse_score / len(test_loader)).detach().numpy())))

    wandb_experiment.log({
        'test set mse': mse_score / len(test_loader),
        'test set rmse': np.sqrt((mse_score / len(test_loader)).detach().numpy())
    })

    for i in range(len(gt)):
        np.save('{}/{}channels_{}_groundtruth_{}.npy'.format(out_dir, channels, target, i), gt[i])
        np.save('{}/{}channels_{}_pred_{}.npy'.format(out_dir, channels, target, i), preds[i])



def predict_probabilistic(in_model, target, test_dataset, wandb_experiment, channels, seed, out_dir, device):
    """
    Predict probabilistic output with uncertainty
    Currently hardcoding for 2016 as test set, will make this
    dynamic
    Need to update this as well because it is outputting garbage atm
    """

    # Make deterministic
    make_deterministic(seed)

    model = models.MCDropoutProbabilisticUNet(n_channels=channels, n_classes=1)
    model.load_state_dict(torch.load(in_model)['state_dict'])

    # Setting model to train mode to retain dropout during inference
    model.train()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    loss_criterion = beta_NLL()
    mse_loss = nn.MSELoss()
    mse_score = 0

    # iterate over the test set
    gt = []
    pred_map_list = []
    epi_unc_list = []
    ale_unc_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sampled_pred_maps, sampled_aleatoric_maps = [], []
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            gt.append(labels.detach().numpy())

            for rep in range(100):
                pred_map_means, pred_map_log_vars = model(inputs)
                sampled_pred_maps.append(pred_map_means.cpu().detach().numpy())
                sampled_aleatoric_maps.append(torch.exp(-pred_map_log_vars).cpu().detach().numpy())

            epi_unc_maps = np.var(np.array(sampled_pred_maps), axis=0)          # Epistemic uncertainty calculated via the variance of the predictions
            ale_unc_maps = np.mean(np.array(sampled_aleatoric_maps), axis=0)
            pred_maps = np.mean(np.array(sampled_pred_maps), axis=0)

            pred_map_list.append(pred_maps)
            epi_unc_list.append(epi_unc_maps)
            ale_unc_list.append(ale_unc_maps)

            # For bias, need to remove nans
            test_mask = ~torch.isnan(labels)

            # Applying mask to remove nans
            no_nan_outputs = pred_maps[test_mask]
            no_nan_outputs_torch = torch.from_numpy(no_nan_outputs)
            labels = labels[test_mask]
            mse_score += mse_loss(no_nan_outputs_torch, labels)

    print('test set mse is: {}'.format(mse_score / len(test_loader)))
    print('test set rmse is: {}'.format(np.sqrt((mse_score / len(test_loader)).detach().numpy())))

    wandb_experiment.log({
        'test set mse': mse_score / len(test_loader),
        'test set rmse': np.sqrt((mse_score / len(test_loader)).detach().numpy())
    })

    total_pred_unc_list = list(map(add, ale_unc_list, epi_unc_list))
    for i in range(len(gt)):
        np.save('{}/{}channels_{}_groundtruth_{}.npy'.format(out_dir, channels, target, i), gt[i])
        np.save('{}/{}channels_{}_pred_{}.npy'.format(out_dir, channels, target, i), pred_map_list[i])
        np.save('{}/{}channels_{}_epi_unc_maps_{}.npy'.format(out_dir, channels, target, i), epi_unc_list[i])
        np.save('{}/{}channels_{}_ale_unc_maps_{}.npy'.format(out_dir, channels, target, i), ale_unc_list[i])
        np.save('{}/{}channels_{}_total_pred_unc_maps_{}.npy'.format(out_dir, channels, target, i),
                total_pred_unc_list[i])