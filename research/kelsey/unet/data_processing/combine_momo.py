"""
Quick script to combine the momo 9 and 7 channel extractions into arrays
"""

import numpy as np
import os

root_momo_dir = '/Users/kelseyd/Desktop/unet/data/NorthAmerica/zscore_normalization/momo'

momo_9_channels = '{}/9_channels'.format(root_momo_dir)
momo_7_channels = '{}/7_channels'.format(root_momo_dir)
momo_16_channels = '{}/16_channels'.format(root_momo_dir)
momo_months = ['june']

def combine_arrays(array1, array2):
    """
    Combine the two arrays and return
    """

    arr_list = []
    # Append first momo array
    for i in range(len(array1[:,0,0])):
        arr_list.append(array1[i, :, :])

    # Append second momo array
    for i in range(len(array2[:, 0, 0])):
        arr_list.append(array2[i, :, :])

    final_array = np.array(arr_list)

    return final_array


for m in momo_months:
    sorted_9_channels = sorted(os.listdir('{}/{}'.format(momo_9_channels,m)))
    sorted_7_channels = sorted(os.listdir('{}/{}'.format(momo_7_channels, m)))
    # to do: iterate through the list of samples, load and combine array via combine arrays function and save
    for samp in range(len(sorted_9_channels)):
        arr9channel = np.load('{}/{}/{}'.format(momo_9_channels, m, sorted_9_channels[samp]))
        arr7channel = np.load('{}/{}/{}'.format(momo_7_channels, m, sorted_7_channels[samp]))
        combined = combine_arrays(arr9channel, arr7channel)
        np.save('{}/{}/{}_sample.npy'.format(momo_16_channels, m, sorted_9_channels[samp][0:10]), combined)





