import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def create_confusion_matrix(model_name, y_pred, targ):
    """
        Parameters
            - model_name: name of the model trained and specify which data is being predicted to create CM
            - y_pred: array with the probability prediction
        Returns
            - shows the confusion matrix plot

    """
    # -- confusion matrix for some data
    con_mat = tf.math.confusion_matrix(labels=targ, predictions=np.argmax(y_pred,axis=1)).numpy()
    con_mat.flatten()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm,
                     index = [0,1], 
                     columns = [0,1])
    con_mat_norm.flatten()
    labels = [f"{v1}\n{v2*100}%" for v1, v2 in
          zip(con_mat.flatten(),con_mat_norm.flatten())]
    labels = np.asarray(labels).reshape(2,2)
    categories = ["0: Real", "1: Bogus"]
    
    figure = plt.figure(figsize=(4, 5))
    sns.heatmap(con_mat, annot=labels,cbar=False,fmt='',xticklabels=categories,yticklabels=categories,cmap='Pastel2_r')#plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusionmatrix_model_%s.pdf"%model_name,bbox_inches="tight")
    plt.show()
    
    return con_mat_df