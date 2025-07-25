"""
Custom metrics for UNet
"""

import numpy as np
import torch

def mse_loss(true, pred):
    """
    Calculate mse loss, ignoring pixels
    with value (hardcoded) == -1000, this
    is missing ground truth from TOAR
    """
    true_arr = true.detach().numpy()
    pred_arr = pred.detach().numpy()

    # iterate through each of the true, pred in batch
    pred_list = []
    true_list = []
    for i in range(true_arr.shape[0]):
        true_i = true_arr[0,0,:,:]
        true_flat = true_i.flatten()
        remove_idx = list(np.where(true_flat==-1000)[0])
        pred_i = pred_arr[0,0,:,:]
        pred_flat = pred_i.flatten()
        pred_no_nan = [i for j, i in enumerate(pred_flat) if j not in remove_idx]
        true_no_nan = [i for j, i in enumerate(true_flat) if j not in remove_idx]
        pred_list.append(pred_no_nan)
        true_list.append(true_no_nan)
    import ipdb
    ipdb.set_trace()
