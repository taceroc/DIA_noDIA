import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd

kernel_size = 5
epochs = 40
batch_size = 20



def keras_model_small_and_save(train, test, train_targ, test_targ ):

    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(16, kernel_size=(kernel_size, kernel_size), padding="valid", activation="relu")
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer3 = keras.layers.Conv2D(32, kernel_size=(kernel_size, kernel_size), padding="valid", activation="relu")
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer5 = keras.layers.Flatten()
    #layer6 = keras.layers.Dropout(0.4)
    layer7 = keras.layers.Dense(32, activation="relu")
    layer8 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1, layer2, layer3, layer4, layer5, layer7,layer8]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    # -- fit the model
    history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=epochs, batch_size=batch_size, verbose = 0)

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ, verbose = 0)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ, verbose = 0)

    # print("Training accuracy : {0:.4f}".format(acc_tr))
    # print("Testing accuracy  : {0:.4f}".format(acc_te))

    model_json = model.to_json()
    with open("model_small.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_small.h5")
    print("Saved model to disk")
    
    return loss_tr, acc_tr, loss_te, acc_te, history, model

def keras_model_AA_and_save(model_name, train, test, train_targ, test_targ, checkpoints = 0):
    """
        Parameters
            - model_name: name of the model to be train 
            - train: array with train data set
            - test: array with test data set
            - train_targ: human label for train data set    
            - test_targ: human label for test data set
            - checkpoints: check if it is a new model (0) or a continuation of another (1)
        Returns
            - history: history of the model
            - model: model to make later model.evaluate or model.predict
            - times: total time required for train the model

    """
    times = 0
    start = 0
    end = 0
    start = time.time()

    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", input_shape=(train.shape[1],train.shape[2],1))
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer3 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation="relu")
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer5 = keras.layers.Conv2D(64, kernel_size=(5, 5), padding="valid", activation="relu")
    layer6 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer7 = keras.layers.Flatten()
    # layer71 = keras.layers.Dropout(0.4)
    layer9 = keras.layers.Dense(32, activation="relu")
    layer10 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer9, layer10]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])


    # -- define the checkpoint
    filepath = "model_checkpoint_%s.h5"%model_name
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                    save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    
    earlyst = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)



    print(train.shape, test.shape)
    print('unique test: {}'.format(np.unique(test_targ, return_counts=True)))
    print('unique train: {}'.format(np.unique(train_targ, return_counts=True)))

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    print(model.summary())

    # -- fit the model 
    if checkpoints == 0:
        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=500, batch_size=200,
                        callbacks=[earlyst, checkpoint])

    # -- specify if there is already a created model, and continue the training
    else:
        model = keras.models.load_model(filepath)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                        save_best_only=True, mode='max')

        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=100, batch_size=200,
                           callbacks=[earlyst, checkpoint])
      

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ, verbose=0)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ, verbose=0)

    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

    end = time.time()
    times  = (end - start)
    print(times)

    # model_json = model.to_json()
    # with open("../outputs/model_%s.json"%model_name, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("../outputs/model_%s.h5"%model_name)
    print("Saved model to disk")
    
    return history, model, times

def keras_model_AAD_and_save(model_name, train, test, train_targ, test_targ, checkpoints = 0):
    """
        Parameters
            - model_name: name of the model to be train 
            - train: array with train data set
            - test: array with test data set
            - train_targ: human label for train data set    
            - test_targ: human label for test data set
            - checkpoints: check if it is a new model (0) or a continuation of another (1)
        Returns
            - history: history of the model
            - model: model to make later model.evaluate or model.predict
            - times: total time required for train the model

    """
    times = 0
    start = 0
    end = 0
    start = time.time()

    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", input_shape=(train.shape[1],train.shape[2],1))
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer21 = keras.layers.Dropout(0.4)
    layer3 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation="relu")
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer41 = keras.layers.Dropout(0.4)
    layer5 = keras.layers.Conv2D(64, kernel_size=(5, 5), padding="valid", activation="relu")
    layer6 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer61 = keras.layers.Dropout(0.4)
    layer7 = keras.layers.Flatten()
    layer9 = keras.layers.Dense(32, activation="relu")
    layer10 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1,layer2,layer21,layer3,layer4,layer41,layer5,layer6,layer61,
              layer7,layer9,layer10]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # -- define the checkpoint
    filepath = "model_checkpoint_%s.h5"%model_name
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                    save_best_only=True, mode='max')
    
    earlyst = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)


    print(train.shape, test.shape)
    print('unique test: {}'.format(np.unique(test_targ, return_counts=True)))
    print('unique train: {}'.format(np.unique(train_targ, return_counts=True)))

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    # -- fit the model 
    if checkpoints == 0:
        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=500, batch_size=200,
                        callbacks=[earlyst, checkpoint])

    # -- specify if there is already a created model, and continue the training
    else:
        model = keras.models.load_model(filepath)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                                        save_best_only=True, mode='min')

        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=250, batch_size=200,
                           callbacks=[earlyst, checkpoint])
    

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ, verbose=0)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ, verbose=0)

    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

    end = time.time()
    times  = (end - start)
    print(times)

    # model_json = model.to_json()
    # with open("../outputs/model_%s.json"%model_name, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("../outputs/model_%s.h5"%model_name)
    # print("Saved model to disk")
    
    return history, model, times

def keras_model_BB_and_save(model_name, train, test, train_targ, test_targ, checkpoints = 0):
    """
        Parameters
            - model_name: name of the model to be train 
            - train: array with train data set
            - test: array with test data set
            - train_targ: human label for train data set    
            - test_targ: human label for test data set
            - checkpoints: check if it is a new model (0) or a continuation of another (1)
        Returns
            - history: history of the model
            - model: model to make later model.evaluate or model.predict
            - times: total time required for train the model

    """
    times = 0
    start = 0
    end = 0
    start = time.time()

    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding="valid", activation="relu", input_shape=(train.shape[1],train.shape[2],1))
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer3 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu")
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer41 = keras.layers.Dropout(0.4)
    layer5 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu")
    layer6 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer8 = keras.layers.Conv2D(128, kernel_size=(3, 3), padding="valid", activation="relu")
    layer81 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer82 = keras.layers.Dropout(0.4)
    layer7 = keras.layers.Flatten()
    layer9 = keras.layers.Dense(32, activation="relu")
    layer10 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1,layer2,layer3,layer4,layer41,layer5,layer6,layer8,layer81,layer82,
              layer7,layer9,layer10]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # -- define the checkpoint
    filepath = "model_checkpoint_%s.h5"%model_name
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                    save_best_only=True, mode='max')
    
    earlyst = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)


    print(train.shape, test.shape)
    print('unique test: {}'.format(np.unique(test_targ, return_counts=True)))
    print('unique train: {}'.format(np.unique(train_targ, return_counts=True)))

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    # -- fit the model 
    if checkpoints == 0:
        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=500, batch_size=200,
                        callbacks=[earlyst, checkpoint])

    # -- specify if there is already a created model, and continue the training
    else:
        model = keras.models.load_model(filepath)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                                        save_best_only=True, mode='min')

        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=250, batch_size=200,
                           callbacks=[earlyst, checkpoint])
    

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ, verbose=0)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ, verbose=0)

    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

    end = time.time()
    times  = (end - start)
    print(times)

    # model_json = model.to_json()
    # with open("../outputs/model_%s.json"%model_name, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("../outputs/model_%s.h5"%model_name)
    # print("Saved model to disk")
    
    return history, model, times

def keras_model_CC_and_save(model_name, train, test, train_targ, test_targ, checkpoints = 0):
    """
        Parameters
            - model_name: name of the model to be train 
            - train: array with train data set
            - test: array with test data set
            - train_targ: human label for train data set    
            - test_targ: human label for test data set
            - checkpoints: check if it is a new model (0) or a continuation of another (1)
        Returns
            - history: history of the model
            - model: model to make later model.evaluate or model.predict
            - times: total time required for train the model

    """
    times = 0
    start = 0
    end = 0
    start = time.time()

    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(1, kernel_size=(7, 7), padding="valid", activation="relu", input_shape=(train.shape[1],train.shape[2],1))
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer3 = keras.layers.Conv2D(16, kernel_size=(3, 3), padding="valid", activation="relu")
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer41 = keras.layers.Dropout(0.4)
    layer5 = keras.layers.Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu")
    layer6 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer82 = keras.layers.Dropout(0.4)
    layer7 = keras.layers.Flatten()
    layer9 = keras.layers.Dense(32, activation="relu")
    layer10 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1,layer2,layer3,layer4,layer41,layer5,layer6,layer82,
              layer7,layer9,layer10]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # -- define the checkpoint
    filepath = "model_checkpoint_%s.h5"%model_name
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                    save_best_only=True, mode='max')
    
    earlyst = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)


    print(train.shape, test.shape)
    print('unique test: {}'.format(np.unique(test_targ, return_counts=True)))
    print('unique train: {}'.format(np.unique(train_targ, return_counts=True)))

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    # -- fit the model 
    if checkpoints == 0:
        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=500, batch_size=200,
                        callbacks=[earlyst, checkpoint])

    # -- specify if there is already a created model, and continue the training
    else:
        model = keras.models.load_model(filepath)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                        save_best_only=True, mode='max')

        history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=250, batch_size=200,
                           callbacks=[earlyst, checkpoint])
    

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ, verbose=0)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ, verbose=0)

    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

    end = time.time()
    times  = (end - start)
    print(times)

    # model_json = model.to_json()
    # with open("../outputs/model_%s.json"%model_name, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("../outputs/model_%s.h5"%model_name)
    # print("Saved model to disk")
    
    return history, model, times




def load_model(model_name, train, test, train_targ, test_targ):
    """
        Loaded previous train CNN model
    """
    print(model_name)
    new_model = keras.models.load_model("model_checkpoint_%s.h5"%model_name)

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), train.shape[1], train.shape[2], 1)
    feat_te2 = test.reshape(len(test), test.shape[1], test.shape[2], 1)

    # -- print the accuracy
    loss_tr, acc_tr = new_model.evaluate(feat_tr2, train_targ, verbose=0)
    loss_te, acc_te = new_model.evaluate(feat_te2, test_targ, verbose=0)
        
    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

    return acc_tr, acc_te, new_model

def predict_data(data, model, created_df = 0, ID = [], targ = [] ):
    """
        Parameters
            - data: data: data to be tested by the CNN model
            - model: CNN model used to predict, return of the funcitons keras_model_and_save or load_model
            - created_df: 
                {
                    0: no generated df with Columns = ["ID", "OBJECT_TYPE", "predicted"]
                    1: oppositie
                }
            - ID: only if created_df = 1 >> IDs for that data
            - targ: only if created_df = 1 >> human labels for that data

        Returns
            {
                created_df = 0
                    y_pred: array with the probability prediction
                created_df = 1
                    - y_pred: array with the probability prediction
                    - df_pred: dataframe with Columns = ["ID", "OBJECT_TYPE", "predicted"], predicted is the integer value
            }
    """
    feat_te2 = data.reshape(len(data), data.shape[1], data.shape[2], 1)
    # -- confusion matrix for some data
    y_pred = model.predict(feat_te2)

    if created_df == 1:
        ID = ID.astype(int)
        targ = targ.astype(int)
        df_pred = pd.DataFrame({'ID':ID, 'OBJECT_TYPE':targ})
        df_pred["predicted_float_Real"] = y_pred[:,0]
        df_pred["predicted_float_Bogus"] = y_pred[:,1]
        df_pred["predicted"] = np.argmax(y_pred,axis=1)
        return df_pred, y_pred
    else:    
        return y_pred
