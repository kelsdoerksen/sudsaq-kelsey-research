"""
Throw-away script to fix arrays to be 9-channel momo
Have identified that momo.2dsfc.NALD variable is one we would
like to ignore, this is channel 4
"""

import numpy as np
import os

data_dir = '/Users/kelseyd/Desktop/unet/data/momo9_and_gee'
save_dir = '/Users/kelseyd/Desktop/unet/data/momo_9'

'''
for f in os.listdir(data_dir):
    arr = np.load('{}/{}'.format(data_dir, f))
    arr_list = []
    for i in range(4):
        arr_list.append(arr[:,:,i])

    for j in range(5,33):
        arr_list.append(arr[:,:,j])

    final_array = np.array(arr_list)
    np.save('{}/{}_momo_modis_pop.npy'.format(save_dir, f[0:10]), final_array)
'''

for f in os.listdir(data_dir):
    arr = np.load('{}/{}'.format(data_dir, f))
    arr_list = []
    for i in range(9):
        arr_list.append(arr[i, :, :])

    final_array = np.array(arr_list)
    np.save('{}/{}_sample.npy'.format(save_dir, f[0:10]), final_array)

