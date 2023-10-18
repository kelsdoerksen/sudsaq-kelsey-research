"""
Script for the losses to use for UQ
From Steven Lu CMB Task
"""

import torch
import torch.nn as nn

class NLL(nn.Module):
  def __init__(self, reduction='mean'):
    super(NLL, self).__init__();

  def forward(self, pred_mean, pred_log_var, target):
    log_var = -pred_log_var
    precision = torch.exp(-log_var)       # also known as variance
    return 0.5 * torch.mean(torch.sum(precision * torch.abs((target - pred_mean))**2 + log_var))


class beta_NLL(nn.Module):
  def __init__(self, beta=0.5, reduction='mean'):
    super(beta_NLL, self).__init__();
    self.beta = beta

  def forward(self, pred_mean, pred_log_var, target):
    precision = torch.exp(-pred_log_var)
    NLL = precision * (target - pred_mean)**2 + pred_log_var
    variance_weighting = torch.exp(pred_log_var) ** self.beta
    betaNLL =  torch.mean(torch.sum(variance_weighting*NLL, 1), 0)
    return betaNLL