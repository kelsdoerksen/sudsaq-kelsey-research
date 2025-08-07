"""
Training script for UNet
Currently have this written to run the whole thing, to update
"""

"""
Note on Gradient Scaling:
If the forward pass for a op have float16 inputs, the backward pass for that op will produce
float16 gradients. Gradient values with small magnitudes may not be representable in float16;
these values will flush to zero (underflow), so the update for the corresponding parameters will be lost.
To combat this, 'gradient scaling' multiplies the networks loss(es) by a scale factor and invokes a backward
pass on the scaled loss(es) -> gradients flowing through the network are scaled to them same factor
"""

from dataset import *
from metrics import *
import numpy as np
from model import *
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, random_split
import wandb
from torch import optim
from pathlib import Path
from predict import *
from losses import *
from utils import *
import random


def train_model(model,
                device,
                dataset,
                save_dir,
                experiment,
                epochs: int,
                batch_size: int,
                learning_rate: float,
                opt,
                val_percent,
                weight_decay: float = 0,
                save_checkpoint: bool=True,
                ):

    # --- Split dataset into training and validation
    seed = random.randint(0, 1000)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # --- DataLoaders
    # The DataLoader pulls instances of data from the Dataset, collects them in batches,
    # and returns them for consumption by your training loop.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # --- Setting up optimizer
    if opt == 'rms':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

    # Setting up loss to filter out nans
    criterion = NoNaNMSE()

    # --- Setting up schedulers
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize MSE score.
    grad_scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    epoch_number = 0
    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        model.train()
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()  # if set_to_none=True, sets gradients of all optimized torch.Tensors to None, will have a lower memory footprint, can modestly improve performance

            outputs = model(inputs)                 # predict on input

            loss = criterion(outputs, labels)       # Calculate loss

            grad_scaler.scale(loss).backward()      # Compute partial derivative of the output f with respect to each of the input variables
            grad_scaler.step(optimizer)             # Updates value of parameters according to strategy
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

        experiment.log({
            'train mse loss': epoch_loss/len(train_loader),
            'train rmse': np.sqrt(epoch_loss/len(train_loader)),
            'step': global_step,
            'epoch': epoch,
            'optimizer': opt
        })

        # Evaluation -> Validation set
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        # Run validation
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        with torch.no_grad():
            for k, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                val_mask = ~torch.isnan(vlabels)
                voutputs = model(vinputs)
                # Filter out nans
                voutputs = voutputs[val_mask]
                vlabels = vlabels[val_mask]
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_loader)
        #scheduler.step(avg_vloss)

        logging.info('Validation MSE score: {}'.format(avg_vloss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation MSE loss': avg_vloss,
                'validation RMSE': np.sqrt(avg_vloss),
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        except:
            pass

        '''
        if save_checkpoint:
            out_model = '{}/checkpoint_epoch{}.pth'.format(save_dir, epoch)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       out_model)
            logging.info(f'Checkpoint {epoch} saved!')
        '''


    # Saving model at end of epoch with experiment name
    out_model = '{}/{}_last_epoch.pth'.format(save_dir, experiment.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               out_model)

    return out_model


def evaluate_probabilistic(model, data_loader, device, num_reps):
    """
    Function to evalauate the probabilistic model during training.
    model: model of interest
    data_loader: data loader to load appropriate data (train or validation)
    num_reps: number of reps to do this for, for mc dropout
    """
    sum_loss = 0.0
    sum_mse = 0.0

    criterion = NLL()
    mse_criterion = nn.MSELoss()
    with torch.no_grad():
        for k, data in enumerate(data_loader):
            model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            sampled_means = []
            sampled_log_vars = []
            mask = ~torch.isnan(labels)
            for rep in range(num_reps):
                means, log_vars = model(inputs)
                sampled_means.append(means.cpu().detach().numpy())
                sampled_log_vars.append(log_vars.cpu().detach().numpy())

            pred_means = np.mean(np.array(sampled_means), axis=0)
            pred_log_vars = np.mean(np.array(sampled_log_vars), axis=0)

            # apply mask
            pred_means = pred_means[mask]
            pred_log_vars = pred_log_vars[mask]
            labels = labels[mask]

            pred_means = torch.from_numpy(pred_means)
            pred_log_vars = torch.from_numpy(pred_log_vars)

            loss = criterion(pred_means, pred_log_vars, labels)
            mse = mse_criterion(pred_means, labels)

            # Add batch loss to total
            sum_loss += loss.item()
            sum_mse += mse.item()

        # Get avg loss which is total loss divided by number of batches == length of data loader
        avg_loss = sum_loss/len(data_loader)
        avg_mse = sum_mse/len(data_loader)

    return avg_loss, avg_mse


def train_probabilistic_model(model,
                              device,
                              dataset,
                              save_dir,
                              experiment,
                              epochs: int,
                              batch_size: int,
                              learning_rate: float,
                              opt,
                              val_percent,
                              weight_decay: float = 1e-3,
                              save_checkpoint: bool=True,
                              ):

    # --- Split dataset into training and validation
    seed = random.randint(0, 1000)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # --- DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # --- Load weights from saved deterministic model -> old, don't need I don't think
    '''
    saved_model_dir = '/Users/kelseyd/Desktop/unet/runs/Europe/mda8/39channels/scarlet-fire-322/checkpoint_epoch199.pth'
    saved_model = torch.load(saved_model_dir)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in saved_model['state_dict'].items() if k in model_dict}
    # Update model with pretrained dict
    model_dict.update(pretrained_dict)
    # Load new state dict values
    model.load_state_dict(model_dict)
    '''

    # --- Initialize small random log_weights
    torch.nn.init.normal_(model.log_var.conv.weight, mean=0.0, std=1e-6)
    torch.nn.init.normal_(model.log_var.conv.bias, mean=0.0, std=1e-6)

    # --- Setting up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # --- Setting up losses
    criterion = NLL()
    mse_criterion = nn.MSELoss()

    # --- Setting up schedulers
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize MSE score.
    grad_scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    epoch_number = 0
    epoch_loss = 0.0
    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        model.train()
        for i, data in enumerate(train_loader):
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

            global_step += 1

            mse_loss = mse_criterion(pred_map_means, labels)
            epoch_loss += loss.item()
            mse_loss += mse_loss.item()

        train_loss, train_mse_loss = evaluate_probabilistic(model, train_loader, device=device, num_reps=5)
        experiment.log({
            'train NLL loss': train_loss,
            'train mse': train_mse_loss,
            'train rmse': np.sqrt(train_mse_loss),
            'step': global_step,
            'epoch': epoch,
            'optimizer': opt
        })

        print('Train NLL for 5 reps: {}'.format(train_loss))
        # Validation
        '''
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
        '''

        # Run validation
        for i, data in enumerate(val_loader):
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

            val_loss = criterion(pred_map_means, pred_map_log_vars, labels)
            val_mse = mse_criterion(pred_map_means, labels)
            val_loss += loss.item()
            val_mse += mse_loss.item()

        val_loss, val_mse = evaluate_probabilistic(model, val_loader, device=device, num_reps=5)

        logging.info('Validation MSE score: {}'.format(val_mse))
        print('Val NLL: {}'.format(val_loss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation NLL loss': val_loss,
                'validation MSE loss': val_mse,
                'validation RMSE': np.sqrt(val_mse),
                'step': global_step,
                'epoch': epoch
            })
        except:
            pass

        '''
        if save_checkpoint:
            out_model = '{}/checkpoint_epoch{}.pth'.format(save_dir, epoch)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       out_model)
            logging.info(f'Checkpoint {epoch} saved!')
        '''

    # Saving model at end of epoch with experiment name
    out_model = '{}/{}_last_epoch.pth'.format(save_dir, experiment.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               out_model)

    return out_model
