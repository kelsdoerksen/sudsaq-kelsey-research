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
from typing import List


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            model.to(device)
            model.train()
            for i, data in enumerate(train_data[model_idx]):
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # Filter out nans to ignore for bias to calculate losses
                mask = ~torch.isnan(labels)

                pred_map_means, pred_map_log_vars = model(inputs)
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
    print('Running inference...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    loss_criterion = NLL()

    pred_mean_list: List[List[torch.Tensor]] = []
    pred_var_list: List[List[torch.Tensor]] = []

    for i, model in enumerate(ensemble):
        model.to(device)
        model.eval()
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                if j >= len(pred_mean_list):
                    pred_mean_list.append([])
                    pred_var_list.append([])

                inputs, labels = inputs.to(device), labels.to(device)
                pred_map_means, pred_map_log_vars = model(inputs)

                pred_mean_list[j].append(pred_map_means.cpu())
                pred_var_list[j].append(pred_map_log_vars.cpu())

    # Now aggregate per batch
    for bidx in range(len(pred_mean_list)):
        means_stack = torch.stack(pred_mean_list[bidx], dim=0)
        logvars_stack = torch.stack(pred_var_list[bidx], dim=0)

        pred_mean = means_stack.mean(dim=0)
        aleatoric = torch.exp(logvars_stack).mean(dim=0)
        epistemic = means_stack.pow(2).mean(dim=0) - pred_mean.pow(2)
        unc_total = aleatoric + epistemic

        # Save per-batch aggregates
        np.save(f'{save_dir}/{channels}channels_{target}_pred_{bidx}.npy',
                pred_mean.detach().cpu().numpy())
        np.save(f'{save_dir}/{channels}channels_{target}_total_pred_unc_map_{bidx}.npy',
                unc_total.detach().cpu().numpy())
        np.save(f'{save_dir}/{channels}channels_{target}_ale_unc_map_{bidx}.npy',
                aleatoric.detach().cpu().numpy())
        np.save(f'{save_dir}/{channels}channels_{target}_epi_unc_map_{bidx}.npy',
                epistemic.detach().cpu().numpy())

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Train models to make up ensemble
    trained_ens = train_ensembles(device, train_dataset, val_percent, channels, epochs, batch_size, learning_rate,
                                  weight_decay, ensemble_size)

    predict_ensembles(trained_ens, test_dataset, device, save_dir, channels, 'bias')