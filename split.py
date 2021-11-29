import numpy as np
import os
import pandas as pd
from config import *
import glob
from astropy.io import fits

pttype = '*'
for ii in range(11):
    
    #Create path for diff, srch, temp images
    print('start {}'.format(ii))
    flist = []
    if ii != 10:
        path = os.path.join(configs["dpath"],'stamps%d'%ii,'SNWG','Archive','*','Y1','*','*',pttype + '*.fits')
        flist.append(sorted(glob.glob(path)))
    else:
        path10 = os.path.join(configs["dpath"],'stamps10','*',pttype + '*.fits')
        flist.append(sorted(glob.glob(path10)))                      
    print(len(flist))
# for i in ["20130829","20130831", "20130901"]:
#     path = os.path.join(configs["dpath"],'stamps1','SNWG','Archive','*','Y1',i,'*',pttype + '*.fits')
#     flist.append(sorted(glob.glob(path)))

    flist = np.concatenate((flist))
    
    ID =[int(f.split('/')[-1][4:-5]) for f in flist]

    #extract from .feather file the ID that are on flist
    ffpath = os.path.join(configs["dpath"], "autoscan_features.3.feather") #this .feather file contain only the ID and OBJECT_TYPE for the images that I have on 
    new_labels = pd.read_feather(ffpath)
    current_labels = new_labels[new_labels["ID"].isin(ID)]
    current_labels = current_labels[["ID", "OBJECT_TYPE"]]
    current_labels.drop_duplicates(inplace=True) 
    current_labels = current_labels.sort_values(by= ["ID"]).reset_index(drop=True)
    counts_type = np.unique(current_labels['OBJECT_TYPE'], return_counts=True)
    #how_many = {"Real (0)":counts_type[1][0], "Bogus (1)": counts_type[1][1] }

    if len(counts_type[0]) == 2:
        print("Real (0) = {} and Bogus (1) = {}".format(counts_type[1][0], counts_type[1][1]))
    if len(counts_type[0]) == 1:
        if counts_type[0] == 0:
            print("Real (0) = {}".format(counts_type[1][0]))
        else:
            print("Bogus (1) = {}".format(counts_type[1][0]))


    imlist_dict = {}

    # stores the name of the images as a list for ID above
    #is a circle because i extract the ID for the flist, buttt
    imlist_dict['flist'] = [f for f in flist if int(f.split('/')[-1][4:-5]) in current_labels['ID'].to_numpy()]
    #print (len(imlist_dict['flist']))
    #print(flist.nbytes)
    #del(flist)
    imlist_dict["imshp"] = fits.open((imlist_dict["flist"][0]))[0].data.shape #shape row,col
    extension="fits"
    imdtype = {"fits":float, "gif":np.uint8, }

    #sort as: descending ID and diff, srch, temp
    imlist_dict["flist"] = sorted(imlist_dict["flist"], key=lambda s: s.split('/')[-1][:4])
    imlist_dict["flist"]= sorted(imlist_dict["flist"], key=lambda s: int(s.split('/')[-1][4:-5]))

    #container for data train and data test
    data_full = np.zeros((len(imlist_dict["flist"]),imlist_dict["imshp"][0], imlist_dict["imshp"][1]),imdtype[extension])

    #fill the container and open images
    for i in range(len(imlist_dict["flist"])):
        datas = fits.open(''.join(imlist_dict["flist"][i]), memmap=True)
        #datas.close()
        data_full[i] = datas[0].data
        #print("{}, path:{}".format(i,imlist_dict["flist"][i]))
        datas.close()

    print(data_full.shape)


    data_norm = data_full.astype(float)
    data_full = None
    # # --normalize
    # # mean and std for diff images
    # # min and max for srch and temp

    data_norm[::3] = (data_norm[::3]- data_norm[::3].mean(axis=(1,2), keepdims=True))/data_norm[::3].std(axis=(1,2), keepdims=True) #diff
    data_norm[1::3]= (data_norm[1::3]-data_norm[1::3].min(axis=(1,2), keepdims=True))/(data_norm[1::3].max(axis=(1,2), keepdims=True)-data_norm[1::3].min(axis=(1,2), keepdims=True)) #srch
    data_norm[2::3]= (data_norm[2::3]-data_norm[2::3].min(axis=(1,2), keepdims=True))/(data_norm[1::3].max(axis=(1,2), keepdims=True)-data_norm[1::3].min(axis=(1,2), keepdims=True)) #temp

    #concatenate diff srch temp for the same ID

    #final_data = np.zeros((int(len(data_full)//3),imlist_dict["imshp"][0], imlist_dict["imshp"][1]*3))
    final_data = np.concatenate((data_norm[::3],data_norm[1::3],data_norm[2::3]), axis = 2)
    data_norm = None
    print('Final lenght of data = {}'.format(final_data.shape)) 

    #exxtract the objects  = 0
    df_ID_0 = current_labels[current_labels["OBJECT_TYPE"]==0]
    #exxtract the objects  = 1
    df_ID_1 = current_labels[current_labels["OBJECT_TYPE"]==1]

    #the len is the minimun of object 0, and object 1. To have equal data of both
    len_each_set = min(len(df_ID_0), len(df_ID_1))
    print(len(df_ID_0), len(df_ID_1))

    if len_each_set != 0:
        if len(df_ID_0) <= len_each_set:
            #extract random the number of data classify as 0
            index_data_ID0 = df_ID_0.sample(len_each_set-10, random_state = 2).sort_index()
            #extract random the number of data classify as 1
            index_data_ID1 = df_ID_1.sample(len_each_set+10,random_state = 2).sort_index()
        else:
            #extract random the number of data classify as 0
            index_data_ID0 = df_ID_0.sample(len_each_set+10, random_state = 2).sort_index()
            #extract random the number of data classify as 1
            index_data_ID1 = df_ID_1.sample(len_each_set-10,random_state = 2).sort_index()
        
    if len(df_ID_0) == 0:
        index_data_ID1 = df_ID_1.sort_index()
        index_data_ID0 = df_ID_0
        finalIDs = index_data_ID1
        #index_data_ID1.to_pickle('ID_stamps%d'%ii+'.pkl')
    if len(df_ID_1) == 0:
        index_data_ID0 = df_ID_0.sort_index()
        index_data_ID1 = df_ID_1
        finalIDs = index_data_ID0
        #index_data_ID0.to_pickle('ID_stamps%d'%ii+'.pkl')

    finalIDs = index_data_ID0.append(index_data_ID1)
    #finalIDs.to_pickle('ID_stamps%d'%ii+'.pkl')

    print(len(index_data_ID1),len(index_data_ID0))

    #convert index to numpy to iterate
    index_ID0 = index_data_ID0.index.to_numpy()

    #convert index to numpy to iterate
    index_ID1 = index_data_ID1.index.to_numpy()

    #concatenate both index
    indexes = sorted(np.concatenate((index_ID0, index_ID1)))

    #extract the data from the index given above, of the complete data, where 0 and 1 are not equal
    equal_type_data = len(indexes)
    print("Len of data where len(ID_0) = len(ID_1) = {}".format(equal_type_data))

    #75% is for training
    #25% testing
    train_len = int(equal_type_data*0.70)
    test_len = equal_type_data  - int(equal_type_data*0.70)
    print('Final lenght of train = {}, Final lenght of test = {} '.format(train_len, test_len))

    import random
    random.seed(4)
    random_index = random.sample(range(0, equal_type_data), train_len)

    train = np.array([final_data[i] for i in [indexes[i] for i in sorted(random_index)]])
    test = np.array([final_data[i] for i in indexes if i not in [indexes[i] for i in sorted(random_index)]])
    
    print(len(train),len(test))
    np.save('../data/data_split_n/train%d'%ii+'.npy', train)
    np.save('../data/data_split_n/test%d'%ii+'.npy', test)
    print('Save train and test for {}'.format(ii))


    # #extracting the label 0 or 1
    targets = [current_labels.iloc[i]["OBJECT_TYPE"] for i in indexes]

    #split the targets
    train_targ = np.array([current_labels.iloc[i]["OBJECT_TYPE"] for i in [indexes[i] for i in sorted(random_index)]])
    test_targ = np.array([current_labels.iloc[i]["OBJECT_TYPE"] for i in indexes if i not in [indexes[i] for i in sorted(random_index)]])

    train_ID = np.array([current_labels.iloc[i]["ID"] for i in [indexes[i] for i in sorted(random_index)]])
    test_ID = np.array([current_labels.iloc[i]["ID"] for i in indexes if i not in [indexes[i] for i in sorted(random_index)]])
    print(len(train_ID),len(test_ID))

    np.save('../data/data_split_n/train_targ_%d'%ii+'.npy', train_targ)
    np.save('../data/data_split_n/test_targ_%d'%ii+'.npy', test_targ)
    print('Save train and test targets for {}'.format(ii))

    np.save('../data/data_split_n/train_ID_%d'%ii+'.npy', train_ID)
    np.save('../data/data_split_n/test_ID_%d'%ii+'.npy', test_ID)
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
