"""
Fig generating for neurips
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Histogram plotting
def plot_hist(gt, pred_sixteen, pred_thirtynine, x_min, name):
    save_dir = '/Users/kelseydoerksen/Desktop/Neurips_Results'
    bins = np.linspace(int(x_min), 60, 500)
    plt.hist(gt, bins, histtype='step', label=['target'])
    plt.hist(pred_sixteen, bins, histtype='step', label=['16-channel prediction'])
    plt.hist(pred_thirtynine, bins, histtype='step', label=['39-channel prediction'])
    #plt.legend(loc='upper left')
    plt.xlabel('Bias')
    plt.ylabel('Count')
    plt.tight_layout()
    #plt.savefig('{}/{}_hist.png'.format(save_dir, name))
    #plt.close()
    plt.show()


unet_dir = '/Users/kelseydoerksen/Desktop/unet/runs/Europe/bias'
rf_dir = '/Users/kelseydoerksen/Desktop/sudsaq_rf/runs/Europe/bias'

# Plotting RF Hist
rf_16 = pd.read_csv('{}/16channels/summer/histogram_data.csv'.format(rf_dir))
rf_39 = pd.read_csv('{}/39channels/summer/histogram_data.csv'.format(rf_dir))
rf_gt = rf_16['gt']
rf_pred_16 = rf_16['pred']
rf_pred_39 = rf_39['pred']

# Plotting UNet Hist
unet_16 = pd.read_csv('{}/16channels/worthy-fire-378/histogram_data.csv'.format(unet_dir))
unet_39 = pd.read_csv('{}/39channels/scarlet-energy-325/histogram_data.csv'.format(unet_dir))
unet_gt = unet_16['gt']
unet_pred_16 = unet_16['pred']
unet_pred_39 = unet_39['pred']

# Plotting UNet high bias hist
#plot_hist(unet_gt, unet_pred_16, unet_pred_39)

# Plotting High Bias UNet
new_df = pd.DataFrame()
new_df['gt'] = unet_gt
new_df['pred_16'] = unet_pred_16
new_df['pred_39'] = unet_pred_39

new_df = new_df[new_df['gt'] > 20]

#plot_hist(rf_gt, rf_pred_16, rf_pred_39, x_min=-60, name='rf')
#plot_hist(unet_gt, unet_pred_16, unet_pred_39, x_min=-60, name='unet')
plot_hist(new_df['gt'], new_df['pred_16'], new_df['pred_39'], x_min=-20, name='unet_high')