"""
Script for the losses to use for UQ
"""

import torch
import torch.nn as nn
import numpy as np


# Adapted from Steven Lu CMB Task
class NLL(nn.Module):
    def __init__(self, reduction='mean'):
        super(NLL, self).__init__();

    def forward(self, pred_mean, pred_log_var, target):
        log_var = -pred_log_var
        precision = torch.exp(-log_var)  # also known as variance
        return 0.5 * torch.mean(torch.sum(precision * torch.abs((target - pred_mean)) ** 2 + log_var))


# Adapted from Steven Lu CMB Task
class beta_NLL(nn.Module):
    def __init__(self, beta=0.5, reduction='mean'):
        super(beta_NLL, self).__init__();
        self.beta = beta

    def forward(self, pred_mean, pred_log_var, target):
        precision = torch.exp(-pred_log_var)
        NLL = precision * (target - pred_mean) ** 2 + pred_log_var
        variance_weighting = torch.exp(pred_log_var) ** self.beta
        betaNLL = torch.mean(torch.sum(variance_weighting * NLL, 1), 0)
        return betaNLL


class PinballLoss(nn.Module):
    def __init__(self, quantile):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)

# Adapted from CQR
class AllQuantileLoss(nn.Module):
    """
    Pinball loss function
    """

    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, preds, target, quantiles):
        """
        Compute pinball loss
        :param: preds: pytorch tensor of estimated labels
        :param: target: pytorch tensor of true labels
        :param: quantiles: list of quantiles
        :return: loss: cost function value
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(quantiles):
          errors = target - preds[:, i]
          losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class NoNaNMSE(nn.Module):
    """
    MSE loss with masking nans
    """
    def __init__(self, reduction='mean'):
        super(NoNaNMSE, self).__init__();

    def forward(self, pred, target):
        # Mask nans first
        mask = ~torch.isnan(target)
        # Add mask before calculating loss to remove nans
        pred = pred[mask]
        target = target[mask]

        return torch.mean(torch.square(pred-target))


class NoNaNPinballLoss(nn.Module):
    def __init__(self, quantile):
        super(NoNaNPinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        # Mask nans first
        mask = ~torch.isnan(y_true)
        # Add mask before calculating loss to remove nans
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        errors = y_true - y_pred
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)


class NoNaNQuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(NoNaNQuantileLoss, self).__init__()
        self.quantile = quantiles

    def forward(self, y_pred, y_true):
        # Mask nans first
        mask = ~torch.isnan(y_true)
        # Add mask before calculating loss to remove nans
        low = y_pred[:, 0, :, :]    # Lower quantile
        med = y_pred[:, 1, :, :]    # Median
        upper = y_pred[:, 2, :, :]  # Upper quantile
        y_true = y_true[mask]
        mask = mask[:, 0, :, :]
        low = low[mask]
        med = med[mask]
        upper = upper[mask]

        errors_low = y_true - low
        errors_med = y_true - med
        errors_upper = y_true - upper

        loss_low = torch.max((0.1-1)*errors_low, 0.1*errors_low)
        loss_med = torch.max((0.5 - 1) * errors_med, 0.5 * errors_med)
        loss_upper = torch.max((0.9 - 1) * errors_upper, 0.9 * errors_upper)

        return torch.mean(loss_low+loss_med+loss_upper)

