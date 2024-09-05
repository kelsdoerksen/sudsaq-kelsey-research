"""
Script
and gee array data for unet
"""
import os
import numpy as np
from numpy import moveaxis
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Combine momochem and GEE data')
    parser.add_argument('--momo_dir', help='MOMOChem Data Directory')
    parser.add_argument('--gee_dir', help='GEE Data Directory')
    parser.add_argument('--save_dir', help='Save Directory')
    parser.add_argument('--region', help='Region')
    parser.add_argument('--month', help='Month')

    return parser.parse_args()


def grab_modis(query_year, gee_data_dir, region):
    """
    Grab correct modis array
    """
    # Load appropriate modis data
    modis_arr = np.load('{}/modis_{}_{}_array.npy'.format(gee_data_dir, query_year, region))

    # Match axis to momo, this is (lat, lon, n_channels)
    #modis_arr = moveaxis(modis_arr, 0, 2)
    #modis_arr = moveaxis(modis_arr, 0, 1)

    return modis_arr


def grab_pop(query_year, gee_data_dir, region):
    """
    Grab correct population array
    5-year dataset, choose one closest to date
    """
    # Pop dict, census data must be from within two years
    pop_dict = {
        '2005': ['2005', '2006', '2007'],
        '2010': ['2008', '2009', '2010', '2011', '2012'],
        '2015': ['2013', '2014', '2015', '2016'],
        '2020': ['2017', '2018', '2019', '2020']
    }

    # Load appropriate pop data based on query year
    census_year = None
    for item in pop_dict.keys():
        if query_year in pop_dict[item]:
            census_year = item

    pop_arr = np.load('{}/population_{}_{}_array.npy'.format(gee_data_dir, census_year, region))

    # Match axis to momo, this is (lat, lon, n_channels)
    #pop_arr = moveaxis(pop_arr, 0, 2)
    #pop_arr = moveaxis(pop_arr, 0, 1)

    return pop_arr

def combine_momo_gee(momo_array, momo_channels, query_year, region, data_dir):
    """
    Combine momo and gee data into one multichannel array
    momo_array: momo_array to generate channels from
    momo_channels: number of momo channels
    query_year: year of processing
    """

    modis_arr = grab_modis(query_year, data_dir, region)
    pop_arr = grab_pop(query_year, data_dir, region)

    arr_list = []
    # Append momo
    for i in range(momo_channels):
        arr_list.append(momo_array[i,:,:])

    # Append modis
    for i in range(19):
        arr_list.append(modis_arr[i,:,:])

    # Append pop
    for i in range(4):
        arr_list.append(pop_arr[i,:,:])


    final_array = np.array(arr_list)
    return final_array


if __name__ == '__main__':
    args = get_args()
    momo_dir = args.momo_dir
    gee_data_dir = args.gee_dir
    month = args.month
    save_dir = args.save_dir
    region = args.region

    momo_channels = 28
    gee_channels = 23
    total_channels = momo_channels + gee_channels

    years = ['2020']
    for y in years:
        print('Generating samples for year, month: {} {}'.format(y, month))
        if int(y) < 2020:
            momo_data_dir = '{}/{}/2005-2019'.format(momo_dir, month)
        else:
            momo_data_dir = '{}/{}/2020'.format(momo_dir, month)

        for f in os.listdir('{}'.format(momo_data_dir)):
            if f[0:4] == y:
                momo = np.load('{}/{}'.format(momo_data_dir, f))
                combined_array = combine_momo_gee(momo, momo_channels, y, region, gee_data_dir)
                np.save('{}/{}/{}_sample.npy'.format(save_dir, month, f[0:10]), combined_array)