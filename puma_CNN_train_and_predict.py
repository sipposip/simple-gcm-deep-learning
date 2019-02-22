
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:25:01 2018

train a CNN for puma. The cnn configuration used is the one obtained
in the tuning process.
At the moment, this script is for a training set which is small
enough to fit into memory in one go.

This is the main experiment, which does the training on the full data, for
different lead times (1-14 days)


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

import keras
from pylab import plt
import tensorflow as tf




from dask.diagnostics import ProgressBar
ProgressBar().register()




# the input file was produced with "puma_CNN_preprocess_inputdata_v2.py"
# it is the full puma output, put into one file, reordered coordinates to
# ('time','lat','lon','lev'), and normalized every level to a mean of 0 and
# a std of 1


#ifile = '/proj/bolinc/users/x_sebsc/simple_gcm_ml/puma/testrun/testrun.reordered.normalized.merged.nc'
#ifile = '/home/s/sebsc/pfs/data/simple_gcm_machinelearning/testrun/testrun.reordered.normalized.merged.nc'
#outdir = '/home/s/sebsc/pfs/data/simple_gcm_machinelearning/tuned/'

ifile='/climstorage/sebastian/plasim/testrun/testrun.reordered.normalized.merged.nc'
outdir='/climstorage/sebastian/simiple_gcm_machinelearning/tuned/'


os.system('mkdir -p '+outdir)


N_gpu = 0

print('open inputdata')

data = xr.open_dataarray(ifile, chunks={'time':1})

# note that the time-variable in the input file is confusing: it contains the
# day of the year of the simulation, thus it is repeating all the time 
# (it loops from 1 to 360 and then jumps back to one)


## define the data

## first define spinup years, this are the first N years
## that will be discarded from the data

spinup_years = 30
N_spinup = 365*spinup_years
data = data.isel(time=slice(N_spinup,None))


# for the tuning, the data was split up into train:dev
# this was not the best choice, as now it gets complicated when increasing the
# amount of training data. therefore, we dont use the training data used in the
# tuning.  we discard it, and then use exactly the same development data, then use the next
# years as test data, and finally the rest as training data

train_years = 100
N_train = 365 * train_years
N_train_usedintuning = 30 * 365
dev_years = 20
test_years=30
N_dev = 365*dev_years
N_test = 365*test_years

x = data

# check that we have enough data for the specifications
if N_train + N_dev + N_test > x.shape[0]:
    raise Exception('not enough timesteps in input file!')


x = x.astype('float32')
x = x[:N_train_usedintuning+N_train+N_dev+N_test]


Nlat,Nlon,n_channels=x.shape[1:4]


def prepare_data(lead_time):
    ''' split up data in predictor and predictant set by shifting
     it according to the given lead time, and then split up
     into train, developement and test set'''
    if lead_time == 0:
        X = x
        y = X[:]
    else:

        X = x[:-lead_time]
        y = x[lead_time:]

    X_train = X[N_train_usedintuning+N_dev+N_test:]
    y_train = y[N_train_usedintuning+N_dev+N_test:]
    
    X_dev = X[N_train_usedintuning:N_train_usedintuning+N_dev]
    y_dev = y[N_train_usedintuning:N_train_usedintuning+N_dev]
    
    X_test = X[N_train_usedintuning+N_dev:N_train_usedintuning+N_dev+N_test]
    y_test = y[N_train_usedintuning+N_dev:N_train_usedintuning+N_dev+N_test]
    
    return X_train,y_train, X_dev, y_dev, X_test, y_test


# fixed (not-tuned params)
batch_size = 32
num_epochs = 10
pool_size = 2
drop_prob=0
conv_activation='relu'


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



#for lead_time in range(1,3):
for lead_time in range(2,15):
    X_train,y_train, X_dev, y_dev, X_test, y_test = prepare_data(lead_time)
    
    
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
    
    
    # load the test data into memory,
    print('loading test data into memory')
    X_test.load()
    y_test.load()
    print('finished loading test data into memory')
    
    
    
    ## the params came out of the tuning process
    params = {'conv_depth': 32, 'hidden_size': 500,
              'kernel_size': 6, 'lr': 0.0001, 'n_hidden_layers': 0}
    
    print(params)
    param_string = '_'.join([str(e) for e in (N_train,num_epochs,lead_time)])



       
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
    
    
    model.save_weights('weights_tunedparams_leadtime'+str(lead_time)+'params_'+param_string+'.h5')
        
        
        # reformat history
        
    hist =  hist.history
    #%%
    ##y_train_predicted = model.predict(X_train)  ## for this we would need to load the whole dataset
    ## into memory, which we cannot do...... one way would be to loop through the train set
    
    
    y_test_predicted = model.predict(X_test)
    
    # y_test_predicted is now a numpy array, but y_test is a xarray dataarray
    # therefore we convert it to an xarray, with exactly the same coordinatas/dims
    # as y_test
    
    y_test_predicted = xr.DataArray(data=y_test_predicted, coords=y_test.coords, dims=y_test.dims)
    
    # save the predictions
    y_test_predicted.to_netcdf(outdir+'/predictions_tuned_leadtime'+str(lead_time)+'params_'+param_string+'.nc')
    y_test.to_netcdf(outdir+'/truevalues_tuned_leadtime'+str(lead_time)+'params_'+param_string+'.nc')

    
    plt.figure()
    plt.plot(hist['val_loss'], label='val_loss')
    plt.plot(hist['loss'], label='train loss')
    
    plt.legend()
    plt.savefig('puma_cnn_history_tunedparams_leadtime'+str(lead_time)+'.svg')
    
    pd.DataFrame(hist).to_csv('history_tunedparams_leadtime'+str(lead_time)+'.csv')
    



