"""
Scratch script to combine momo
and gee array data for unet
"""
import os
import numpy as np
from numpy import moveaxis

momo_data_dir = '/Users/kelseyd/Desktop/unet/data/NorthAmerica/zscore_normalization'
gee_data_dir = '/Users/kelseyd/Desktop/unet/data/NorthAmerica/zscore_normalization/gee'


def grab_modis(query_year):
    """
    Grab correct modis array
    """
    # Load appropriate modis data
    modis_arr = np.load('{}/modis/{}_modis_array.npy'.format(gee_data_dir, query_year))

    # Match axis to momo, this is (lat, lon, n_channels)
    modis_arr = moveaxis(modis_arr, 0, 2)
    modis_arr = moveaxis(modis_arr, 0, 1)

    return modis_arr

def grab_pop(query_year):
    """
    Grab correct population array
    5-year dataset, choose one closest to date
    """
    # Pop dict, census data must be from within two years
    pop_dict = {
        '2005': ['2005', '2006', '2007'],
        '2010': ['2008', '2009', '2010', '2011', '2012'],
        '2015': ['2013', '2014', '2015', '2016']
    }

    # Load appropriate pop data based on query year
    census_year = None
    for item in pop_dict.keys():
        if query_year in pop_dict[item]:
            census_year = item

    pop_arr = np.load('{}/population/{}_population_array.npy'.format(gee_data_dir, census_year))

    # Match axis to momo, this is (lat, lon, n_channels)
    pop_arr = moveaxis(pop_arr, 0, 2)
    pop_arr = moveaxis(pop_arr, 0, 1)

    return pop_arr

def combine_momo_gee(momo_array, momo_channels, query_year):
    """
    Combine momo and gee data into one multichannel array
    momo_array: momo_array to generate channels from
    momo_channels: number of momo channels (currently supports 7, 9)
    query_year: year of processing
    """

    modis_arr = grab_modis(query_year)
    pop_arr = grab_pop(query_year)

    arr_list = []
    # Append momo
    for i in range(momo_channels):
        arr_list.append(momo_array[i,:,:])

    # Append modis
    for i in range(19):
        arr_list.append(modis_arr[:,:,i])

    # Append pop
    for i in range(4):
        arr_list.append(pop_arr[:,:,i])

    final_array = np.array(arr_list)
    final_array = moveaxis(final_array, 0, 2)
    return final_array

momo_channels = 9
gee_channels = 23
years = ['2005','2006','2007','2008', '2009','2010','2011','2012','2013','2014','2015','2016']
momo_months = ['june']

total_channels = momo_channels + gee_channels
for y in years:
    print('Generating samples for year: {}'.format(y))
    for m in momo_months:
        print('Generating samples for month: {}'.format(m))
        combined_data_dir = '/Users/kelseyd/Desktop/unet/data/NorthAmerica/zscore_normalization/{}_channels/{}'. \
            format(total_channels, m)
        for f in os.listdir('{}/{}_channels/{}'.format(momo_data_dir, momo_channels, m)):
            if f[0:4] == y:
                momo = np.load('{}/{}_channels/{}/{}'.format(momo_data_dir, momo_channels, m, f))
                combined_array = combine_momo_gee(momo, momo_channels, y)
                np.save('{}/{}_{}channels.npy'.format(combined_data_dir, f[0:10], total_channels), combined_array)