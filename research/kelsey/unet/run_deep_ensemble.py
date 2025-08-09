"""
Run deep ensemble train and test
"""

import random
import torch
from model import *
from dataset import *
from metrics import *
import numpy as np
from model import *
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, random_split
import wandb
from torch import optim
from pathlib import Path
from predict import *
from losses import *
from utils import *


def train_ensembles(device,
                    train_dataset,
                    val_percent,
                    channels,
                    epochs: int,
                    batch_size: int,
                    learning_rate: float,
                    weight_decay: float=0,
                    ensemble_size: int=10,
                    ):
    """
    Train ensemble models
    """

    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    criterion = NLL()

    ensemble = []
    optimizers = []
    train_data = []
    val_data = []
    for i in range(ensemble_size):
        seed = random.randint(0, 1000)
        torch.manual_seed(seed)
        train_set, val_set = random_split(train_dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(seed))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        unet = models.ProbabilisticUNet(n_channels=channels, n_classes=1)
        optimizer = optim.Adam(unet.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
        ensemble.append(unet)
        optimizers.append(optimizer)
        train_data.append(train_loader)
        val_data.append(val_loader)

    for epoch in range(epochs):
        for model_idx, model in enumerate(ensemble):
            optimizer = optimizers[model_idx]
            model.train()
            for i, data in enumerate(train_data[model_idx]):
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero gradients for every batch
                optimizer.zero_grad()
                pred_map_means, pred_map_log_vars = model(inputs)
                # Filter out nans to ignore for bias to calculate losses
                mask = ~torch.isnan(labels)
                pred_map_means = pred_map_means[mask]
                labels = labels[mask]
                pred_map_log_vars = pred_map_log_vars[mask]

                loss = criterion(pred_map_means, pred_map_log_vars, labels)
                loss.backward()
                optimizer.step()

            for i, data in enumerate(val_data[model_idx]):
                vinputs, vlabels = data
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                # Zero gradients for every batch
                optimizer.zero_grad()

                vpred_map_means, vpred_map_log_vars = model(vinputs)

                # Filter out nans to ignore for bias to calculate losses
                vmask = ~torch.isnan(vlabels)
                vpred_map_means = vpred_map_means[vmask]
                vlabels = vlabels[vmask]
                vpred_map_log_vars = vpred_map_log_vars[vmask]

                val_loss = criterion(vpred_map_means, vpred_map_log_vars, vlabels)

    return ensemble


def predict_ensembles(ensemble,
                      test_dataset,
                      device,
                      save_dir,
                      channels,
                      target='bias'):
    """
    Run prediction on ensemble models
    """

    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    loss_criterion = NLL()

    # iterate over the test set
    for i in range(len(ensemble)):
        model = ensemble[i]
        gt = []
        pred_mean_list = []
        pred_var_list = []
        nll_score = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                gt.append(labels.detach().numpy())

                pred_map_means, pred_map_log_vars = model(inputs)
                pred_mean_list.append(pred_map_means.detach().numpy())
                pred_var_list.append(pred_map_log_vars.detach().numpy())

                # For bias, need to remove nans
                test_mask = ~torch.isnan(labels)

                # Applying mask to remove nans
                no_nan_outputs_means = pred_map_means[test_mask]
                no_nan_outputs_log_vars = pred_map_log_vars[test_mask]
                labels = labels[test_mask]
                nll_score += loss_criterion(no_nan_outputs_means, no_nan_outputs_log_vars, labels)

        for i in range(len(gt)):
            np.save('{}/{}channels_{}_groundtruth_{}.npy'.format(save_dir, channels, target, i), gt[i])
            np.save('{}/{}channels_{}_pred_mean_{}.npy'.format(save_dir, channels, target, i), pred_mean_list[i])
            np.save('{}/{}channels_{}_pred_var_{}.npy'.format(save_dir, channels, target, i), pred_var_list[i])

    return



def run_deep_ensemble(device,
                      train_dataset,
                      test_dataset,
                      val_percent,
                      save_dir,
                      channels,
                      epochs: int,
                      batch_size: int,
                      learning_rate: float,
                      weight_decay: float=0,
                      ensemble_size: int=10):
    """
    Running deep ensemble:
    Load model with different seed
    train
    predict
    average
    """

    # Train models to make up ensemble
    trained_ens = train_ensembles(device, train_dataset, val_percent, channels, epochs, batch_size, learning_rate,
                                  weight_decay, ensemble_size)

    predict_ensembles(trained_ens, test_dataset, device, save_dir, channels, 'bias')