import data_utils
import model_utils
import plot_utils
import matplotlib.pyplot as plt
import sys
import os

def multiple_models(data_location, model_name, ddh = 0, checkpoints = 0):
    
    if checkpoints == 1:
        comd = "cp model_checkpoint_%s.h5 model_checkpoint_%s_copy.h5"%(model_name, model_name)
        os.system(comd)
        
    train, test, train_targ, test_targ, train_ID, test_ID = data_utils.read_data_folders(data_location, 10000,
                                                                          1, ddh = ddh)
    print(train.shape, test.shape)
    if ddh == 0:
        history, model, times = model_utils.keras_model_AAD_and_save(model_name, train, test, 
                                                train_targ, test_targ, checkpoints = checkpoints)
    else:
        history, model, times = model_utils.keras_model_CC_and_save(model_name, train, test, 
                                                train_targ, test_targ, checkpoints = checkpoints)
#         history, model, times = keras_model_BB_and_save(model_name, train, test, 
#                                                 train_targ, test_targ, checkpoints = checkpoints)

    fig, ax = plt.subplots(2,1,figsize=(8, 8))
    ax[0].plot(history.history["loss"])
    ax[0].plot(history.history["val_loss"])
    ax[0].legend(["train", "test"], loc="upper left")
    ax[0].set_xlabel("epoch", fontsize=15)
    ax[0].set_ylabel("loss", fontsize=15)

    ax[1].plot(history.history["accuracy"])
    ax[1].plot(history.history["val_accuracy"])
    ax[1].legend(["train", "test"], loc="upper left")
    ax[1].set_xlabel("epoch", fontsize=15)
    ax[1].set_ylabel("acc", fontsize=15)
    if checkpoints == 0:
        plt.savefig("../outputs/lossacc_model%s.pdf"%model_name,bbox_inches="tight")
    else:
        plt.savefig("../outputs/lossacc_model%s_checkpoint.pdf"%model_name, bbox_inches="tight")
    
    y_pred_test = model_utils.predict_data(test, model, 0, test_ID,test_targ)
    if checkpoints == 0:
        plot_utils.create_confusion_matrix("test_%s"%model_name, y_pred_test, test_targ)
    else:
        plot_utils.create_confusion_matrix("test_%s_checkpoint"%model_name, y_pred_test, test_targ)

    y_pred_train = model_utils.predict_data(train, model, 0, train_ID, train_targ)
    if checkpoints == 0:
        plot_utils.create_confusion_matrix("train_%s"%model_name, y_pred_train, train_targ)
    else:
        plot_utils.create_confusion_matrix("train_%s_checkpoint"%model_name, y_pred_train, train_targ)
    
    return train, test, train_targ, test_targ, train_ID, test_ID


if __name__ == "__main__":
    name = sys.argv[1]
    model_name = name
    print(model_name)
    div = name.split("_")[1]
    if div[0:2] == "3D" or div[0:2] == "2D":
        data_location = "../data/data_split/"
    if div[0:2] == "n3" or div[0:2] == "n2":
        data_location = "../data/data_split_n/"
    if div[0:2] == "3s":
        data_location = "../data/data_split_3s/"
    print(data_location)
    #data_location = "../data/data_split_n/"
    if div[-3:] == "3DH":
        ddhs = 0
    if div[-3:] == "2DH":
        ddhs = 1
        
    train, test, train_targ, test_targ, train_ID, test_ID = multiple_models(data_location, model_name, ddh = ddhs, checkpoints = 1)
