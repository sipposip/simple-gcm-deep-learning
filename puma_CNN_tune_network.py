#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:25:01 2018

tuning of CNN for puma. This script is for a training set which is small
enough to fit into memory in one go.


@author: sebastian
"""
import matplotlib
matplotlib.use('agg')

import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
from keras import layers
from keras import backend as K
from keras import optimizers
import keras
from pylab import plt
import tensorflow as tf


from sklearn.model_selection import ParameterGrid


from dask.diagnostics import ProgressBar
ProgressBar().register()


#ifile = '/proj/bolinc/users/x_sebsc/simple_gcm_ml/puma/testrun/testrun.reordered.normalized.merged.nc'
ifile = '/home/s/sebsc/pfs/data/simple_gcm_machinelearning/testrun/testrun.reordered.normalized.merged.nc'
#ifile='/climstorage/sebastian/plasim/testrun/testrun.reordered.normalized.merged.nc'

print('open inputdata')

data = xr.open_dataarray(ifile, chunks={'time':1})



spinup_years = 30
N_spinup = 365*spinup_years
data = data.isel(time=slice(N_spinup,None))


train_years = 30
N_train = 365 * train_years
dev_years = 20
N_dev = 365*dev_years
lead_time=4
x = data


x = x.astype('float32')
x = x[:N_train+N_dev]


Nlat,Nlon,n_channels=x.shape[1:4]


def prepare_data(lead_time):
    ''' split up data in predictor and predictant set by shifting
     it according to the given lead time, and then split up
     into train and developement set'''
    if lead_time == 0:
        X = x
        y = X[:]
    else:

        X = x[:-lead_time]
        y = x[lead_time:]

    X_train = X[:-N_dev]
    y_train = y[:-N_dev]
    
    X_dev = X[-N_dev:]
    y_dev = y[-N_dev:]
    
    return X_train,y_train, X_dev, y_dev


# fixed (not-tuned params)
batch_size = 32
num_epochs = 5
pool_size = 2
drop_prob=0
conv_activation='relu'

param_string = '_'.join([str(e) for e in (N_train,lead_time,batch_size,num_epochs,pool_size,drop_prob)])

N_gpu = 2

def acc_score(x,y):
    '''timestepwise anomaly correlation coefficient, averaged over time
        (simple version without seasonal climatoloty)'''
    assert(x.shape==y.shape)
    return np.mean([np.corrcoef(x[i].flatten(),y[i].flatten())[0,1] for i in range(len(x))])
    
    


def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([
            
            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlat,Nlon,n_channels)),
            layers.MaxPooling2D(pool_size=pool_size),
            Dropout(drop_prob),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.MaxPooling2D(pool_size=pool_size),
            # end "encoder"
            
            
            # dense layers (flattening and reshaping happens automatically)
            ] + [layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +
             
            [
            
            
            # start "Decoder" (mirror of the encoder above)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.UpSampling2D(size=pool_size),
            layers.Convolution2D(n_channels, kernel_size, padding='same', activation=None)
            ]
            )
    
    
    optimizer= keras.optimizers.adam(lr=lr)

    if N_gpu > 1:
        with tf.device("/cpu:0"):
            # convert the model to a model that can be trained with N_GPU GPUs
             model = keras.utils.multi_gpu_model(model, gpus=N_gpu)
             
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    
    return model





X_train,y_train, X_dev, y_dev = prepare_data(lead_time)


# load the devolepment data into memory,
print('loading dev data into memory')
X_dev.load()
y_dev.load()
print('finished loading test data into memory')

# load the train data into memory,
print('loading train data into memory')
X_train.load()
y_train.load()
print('finished loading train data into memory')


tunable_params = dict(
                  lr=[0.00001,0.00003,0.0001],
                  n_hidden_layers = [0,1,2],
                  kernel_size = [2,4,6],
                  hidden_size=[50,100,300,500],
                  conv_depth = [16,32]
        )

    
print(lead_time)


# loop over all tunable parameter configurations. This can be done with scikit
# learns ParameterGrid

param_grid = list(ParameterGrid(tunable_params))
n_param_combis = len(param_grid)
print('trying ',n_param_combis,' combinations')

res = []
for i,params in enumerate(param_grid):
    print('training on param set ',i)
    print(params)
    
   
    model = build_model(**params)
    
    print(model.summary())
    

    print('start training')
    hist = model.fit(X_train, y_train,
                       batch_size = batch_size,
             verbose=1, 
             epochs = num_epochs,
             validation_data=(X_dev,y_dev),
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights.h5', monitor='val_loss', 
                                                    verbose=1, save_best_only=True, 
                                                    save_weights_only=True, mode='auto', period=1)]
             )
    
    print('finished training')
    # get best model from the training (based on validation loss),
    # this is neccessary because the early stopping callback saves the model "patience" epochs after the best one
    model.load_weights('best_weights.h5')
    
    # remove the file created by ModelCheckppoint
    os.system('rm best_weights.h5')
    

    model.save_weights('weights_'+param_string+'_tuning_paramset'+str(i)+'.h5')
    
    
    # reformat history
    
    hist =  hist.history
    #%%
    #y_train_predicted = model.predict(X_train)  ## for this we would need to load the whole dataset
    #3 into memroy, which we cannot do...... one way would be to loop through the train set
    
    
    y_dev_predicted = model.predict(X_dev)
    
    
    # compute scores
    rmse = np.sqrt(np.mean((y_dev_predicted - y_dev.values)**2))
    acc = acc_score(y_dev_predicted, y_dev.values)

    res.append(dict(hist=hist,params=params, scores=[rmse,acc]))
    
    plt.figure()
    plt.plot(hist['val_loss'], label='val_loss')
    plt.plot(hist['loss'], label='train loss')
    
    plt.legend()
    plt.savefig('puma_cnn_history'+param_string+'_tuning_paramset'+str(i)+'.svg')
    
    pd.DataFrame(hist).to_csv('history_'+param_string+'_tuning_paramset'+str(i)+'.csv')
    


pickle.dump(res,open('tuning_result'+param_string+'.pkl','wb'))
