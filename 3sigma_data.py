import numpy as np
import os
import pandas as pd
from config import *
import glob
from astropy.io import fits
import matplotlib.pyplot as plt

import bogus
import data_utils
import open_data
import data_split


def compressed_3sgima(data):
    upper = data < data.mean(axis=(1,2), keepdims=True) + 3*data.std(axis=(1,2), keepdims=True)
    lower = data > data.mean(axis=(1,2), keepdims=True) - 3*data.std(axis=(1,2), keepdims=True)
    mini = np.zeros((len(data),data[0].shape[0],data[0].shape[1]))
    maxi = np.zeros((len(data),data[0].shape[0],data[0].shape[1]))
    for i in range(len(data)):
        thrup = data[i][upper[i] == lower[i]]
        minis = thrup.min()
        maxis = thrup.max()
        mini[i] = np.full((data[0].shape[0],data[0].shape[1]), minis)
        maxi[i] = np.full((data[0].shape[0],data[0].shape[1]), maxis)
    return mini, maxi


for ii in range(11):
    print('start {}'.format(ii))
    flist = data_utils.created_path(ii)
    ID = data_utils.return_ids(flist)
    current_labels = data_utils.DF_current_ids(ID)
    data_full = open_data.open_fits(current_labels, flist)
    data_norm = data_full.astype(float)
    data_norm[::3] = (data_norm[::3]- data_norm[::3].mean(axis=(1,2), keepdims=True))/data_norm[::3].std(axis=(1,2), keepdims=True) #diff
    mini1, maxi1 = compressed_3sgima(data_norm[1::3])    
    data_norm[1::3] = (data_full[1::3] - mini1) / (maxi1-mini1)  
    mini2, maxi2 = compressed_3sgima(data_norm[2::3])    
    data_norm[2::3] = (data_full[2::3] - mini2) / (maxi2-mini2)
    final_data = open_data.concatenate_normdata(data_norm)  
    df_ID_0, df_ID_1 = data_utils.separate_types(current_labels)
    indexes, equal_type_data = data_split.balance_data(df_ID_0, df_ID_1, final_data)
    train, test, random_index = data_split.split_data(equal_type_data, final_data, indexes)
    print(len(train),len(test))
    np.save('../data/data_split_3s/train%d'%ii+'.npy', train)
    np.save('../data/data_split_3s/test%d'%ii+'.npy', test)
    print('Save train and test for {}'.format(ii))
    train_targ, test_targ = data_utils.targets_train_test(current_labels, indexes, random_index)
    train_ID, test_ID = data_utils.ID_train_test(current_labels, indexes, random_index)
    np.save('../data/data_split_3s/train_targ_%d'%ii+'.npy', train_targ)
    np.save('../data/data_split_3s/test_targ_%d'%ii+'.npy', test_targ)
    print('Save train and test targets for {}'.format(ii))

    np.save('../data/data_split_3s/train_ID_%d'%ii+'.npy', train_ID)
    np.save('../data/data_split_3s/test_ID_%d'%ii+'.npy', test_ID)
    print('Save train and test IDs for {}'.format(ii))
    
    (unique, counts) = np.unique(test_targ, return_counts=True)
    print(unique, counts)

    (unique, counts) = np.unique(train_targ, return_counts=True)
    print(unique, counts)
    
    print('Done with {}'.format(ii))
    flist = None
    final_data = None
    train = None
    test = None
    imlist_dict = None