import numpy as np
from astropy.io import fits


def open_fits(current_labels, flist):
    """
        Input:
            - current_labels: dataframe with Columns = ["ID", "OBJECT_TYPE"] where "ID" is the unique IDs and OBJECT_TYPE is the human label ("0": Real, "1": Bogus)
            - flist: The path of all the images, the return of the function created_path()
        Returns:
            data_full: np.array with all the data saved as a array of numbers. The opened data must correspond with the IDs on the data frame of current_labels

    """
    # -- created a dictonary to stored the data reported on the data frame of current_labels
    imlist_dict = {}
    
    # -- stores the name of the images as a list for each ID 
    imlist_dict['flist'] = [f for f in flist if int(f.split('/')[-1][4:-5]) in current_labels['ID'].to_numpy()]
    # -- open the first image and extract the shape
    imlist_dict["imshp"] = fits.open((imlist_dict["flist"][0]))[0].data.shape #shape row,col
    extension="fits"
    imdtype = {"fits":float, "gif":np.uint8}
    
    # -- sort as: diff, srch, temp
    imlist_dict["flist"] = sorted(imlist_dict["flist"], key=lambda s: s.split('/')[-1][:4])
    # -- sort as: descending ID, convert float to integer 
    imlist_dict["flist"]= sorted(imlist_dict["flist"], key=lambda s: int(s.split('/')[-1][4:-5]))
    
    # -- container for data train and data test (total_data, rows, cols)
    data_full = np.zeros((len(imlist_dict["flist"]),imlist_dict["imshp"][0], imlist_dict["imshp"][1]),imdtype[extension])

    # -- fill the container and open images
    for i in range(len(imlist_dict["flist"])):
        datas = fits.open(''.join(imlist_dict["flist"][i]), memmap=True)
        data_full[i] = datas[0].data
        datas.close()
    del(imlist_dict)
    return data_full


def norm_data(data_full):
    """
        Input:
            data_full: np.array with all the data saved as a array of numbers. The opened data must correspond with the IDs on the data frame of current_labels
        Returns:
            data_norm: normalized data, shape = (Total_images, rows, cols)

    """
    # -- convert data to floats
    data_norm = data_full.astype(float)
    data_full = None
    # -- normalize
    # -- mean and std for diff images
    data_norm[::3] = (data_norm[::3]- data_norm[::3].mean(axis=(1,2), keepdims=True))/data_norm[::3].std(axis=(1,2), keepdims=True) #diff
    # -- min and max for srch and temp
    data_norm[1::3]= (data_norm[1::3]-data_norm[1::3].min(axis=(1,2), keepdims=True))/(data_norm[1::3].max(axis=(1,2), keepdims=True)- data_norm[1::3].min(axis=(1,2), keepdims=True)) #srch

    data_norm[2::3]= (data_norm[2::3]-data_norm[2::3].min(axis=(1,2), keepdims=True))/data_norm[2::3].max(axis=(1,2), keepdims=True)-data_norm[2::3].min(axis=(1,2)) #temp

    return data_norm


def sigma3_norm_data(data_full):
    """
        Input:
            data_full: np.array with all the data saved as a array of numbers. The opened data must correspond with the IDs on the data frame of current_labels
        Returns:
            data_norm: normalized data using the 3sigma interval, shape = (Total_images, rows, cols)

    """
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


    # -- convert data to floats
    data_norm = data_full.astype(float)
    data_full = None
    # -- normalize
    # -- mean and std for diff images

    data_norm[::3] = (data_norm[::3]- data_norm[::3].mean(axis=(1,2), keepdims=True))/data_norm[::3].std(axis=(1,2), keepdims=True) #diff

    mini1, maxi1 = compressed_3sgima(data_norm[1::3])    
    data_norm[1::3] = (data_norm[1::3] - mini1) / (maxi1-mini1)  
    mini2, maxi2 = compressed_3sgima(data_norm[2::3])    
    data_norm[2::3] = (data_norm[2::3] - mini2) / (maxi2-mini2)

    return data_norm



def concatenate_normdata(data_norm):
    """
        Input:
            data_norm: normalized data
        Returns:
            final_data: concatenate horizontally diff srch temp for the same ID, shape (Count of IDs, rows, cols*3)
    """
    # -- concatenate diff srch temp for the same ID
    #final_data = np.zeros((int(len(data_norm)//3),data_norm.shape[1], data_norm.shape[1]*3))
    final_data = np.concatenate((data_norm[::3],data_norm[1::3],data_norm[2::3]), axis = 2)
    return final_data