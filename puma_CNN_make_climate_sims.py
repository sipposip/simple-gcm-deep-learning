
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


plt.rcParams['savefig.bbox'] = 'tight'

# the input file was produced with "puma_CNN_preprocess_inputdata_v2.py"
# it is the full puma output, put into one file, reordered coordinates to
# ('time','lat','lon','lev'), and normalized every level to a mean of 0 and
# a std of 1


#ifile = '/proj/bolinc/users/x_sebsc/simple_gcm_ml/puma/testrun/testrun.reordered.normalized.merged.nc'
#ifile = '/home/s/sebsc/pfs/data/simple_gcm_machinelearning/testrun/testrun.reordered.normalized.merged.nc'
#outdir = '/home/s/sebsc/pfs/data/simple_gcm_machinelearning/tuned/'

ifile='/climstorage/sebastian/plasim/testrun/testrun.reordered.normalized.merged.nc'


# the level and variable dimension is the same, it is ua100hPa, ua200hPa,....ua10000hPa, va100hPa....]
varnames = ['ua','va','ta','zg']
keys = [ varname+str(lev) for varname in varnames for lev in range (100,1001,100)]
varname_to_levidx = { key:levidx for key,levidx in zip(keys,range(len(keys)))  }




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


lead_time = 1

# we need tha original model data for initialization, for this we use
# the (unseen) test data

X_train,y_train, X_dev, y_dev, X_test, y_test = prepare_data(lead_time)
    

    
# load the test data into memory,
print('loading test data into memory')
X_test.load()

    
    
# now we load the trianed network. 
# we did not save the complete network, but only the weights.
# therefore we first need to built the network (exactly the same way as it
# was saved)
    
    ## the params came out of the tuning process
params = {'conv_depth': 32, 'hidden_size': 500,
          'kernel_size': 6, 'lr': 0.0001, 'n_hidden_layers': 0}

print(params)
param_string = '_'.join([str(e) for e in (N_train,num_epochs,lead_time)])



   
model = build_model(**params)

print(model.summary())
    
# now load the weights  
weights_file =  '/home/sebastian/simple_gcm_machinelearning/plasim/tuned/weights_tunedparams_leadtime'+str(lead_time)+'params_'+param_string+'.h5'

model.load_weights(weights_file)



N_years = 30

# the number of forecast steps to make depends on the lead_time

N_steps = int(np.round(N_years * 365 / lead_time))
# now we need to select an initial condition, and then repeatedly 
# apply the prediction    

for count_init, i_init in enumerate(np.random.randint(0,len(X_test), size=20)):


    
    initial_state = X_test[i_init]    
    clim_run = np.zeros(shape=[N_steps+1] + list(initial_state.shape))
    
    clim_run = xr.DataArray(data = clim_run, coords={'time':np.arange(N_steps+1),
                                                     'lat':X_test.lat,
                                                     'lon':X_test.lon,
                                                     'lev':X_test.lev},dims=X_test.dims
        )
    
    clim_run[0] = initial_state
    for i in range(N_steps):
        print(i,N_steps)
        # we have to add an (empty) time dimenstion
        current_state = np.expand_dims(clim_run[i],0)
        prediction = model.predict(current_state)
        clim_run[i+1] = np.squeeze(prediction)
    
    
    
    # compute statistics of the prediction
    climmean = clim_run.mean('time')
    
    var='zg500'

    plt.figure(figsize=(7,3))
    climmean.isel(lev=varname_to_levidx[var]).plot.contourf(
            levels=np.arange(-2,2.01,0.1)
            )
    plt.title('network 500hPa height  mean')
    plt.savefig('netowrk_climmean'+str(lead_time)+'.pdf')
    
    
    # plot the climatology of the model
    climmean_puma = X_test.mean('time')
    
    plt.figure(figsize=(7,3))
    climmean_puma.isel(lev=varname_to_levidx[var]).plot.contourf(
            levels=np.arange(-2,2.01,0.1)
            )
    plt.title('gcm 500hPa height mean')
    plt.savefig('puma_climmean'+str(lead_time)+'.pdf')
    
    
    
    # plot one gridpoint
    var='zg500'
    
    plt.figure()
    plt.plot(clim_run[:100,10,10,varname_to_levidx[var]], label='network')
    plt.plot(X_test[i_init:i_init+100*lead_time:lead_time,10,10,varname_to_levidx[var]], label='puma')
    plt.legend()
    sns.despine()
    plt.savefig('timeevolution_one_gridpoint_climatemode'+str(lead_time)+'_init'+str(count_init)+'.pdf')
    
    
    
    
    
    #%% compute variances
    ## this is a bit tricky if the lead_time of the network is not 1 day
    
    # lead_time=1 case
    
    clim_run['time'] = pd.date_range('20000101',pd.to_datetime('20000101')+pd.Timedelta(str(N_steps*lead_time)+'d'), freq=str(lead_time)+'d')
    
    X_test['time'] = pd.date_range('20000101',pd.to_datetime('20000101')+pd.Timedelta(str(len(X_test)-1)+'d'))
    
    for aggregation in (1,10,30):
            clim_newtork_agg = clim_run.resample(time=str(aggregation)+'d').mean('time')
            puma_agg = X_test.resample(time=str(aggregation)+'d').mean('time')
            
            
            network_variance = clim_newtork_agg.var('time')
            puma_variance = puma_agg.var('time')
            
            var='zg500'
            
            if aggregation == 30:
                vmax = 0.002
            elif aggregation == 1:
                vmax = 0.06
            elif aggregation == 10:
                vmax = 0.01
                
            vmin = 0   
            cmap=plt.cm.gist_heat_r
            plt.figure(figsize=(7,3))
            network_variance.isel(lev=varname_to_levidx[var]).plot.contourf(vmin=vmin,vmax=vmax,
                    #levels=np.arange(-2,2.01,0.1)
                    cmap=cmap
                    )
            plt.title('network 500hPa height '+str(aggregation)+'day variance')
            plt.savefig('newtork_'+str(aggregation)+'dayvariance_leadtime'+str(lead_time)+'_init'+str(count_init)+'.pdf')
            
            plt.figure(figsize=(7,3))
            puma_variance.isel(lev=varname_to_levidx[var]).plot.contourf(vmin=vmin,vmax=vmax,
                    #levels=np.arange(-2,2.01,0.1)
                    cmap=cmap
                    )
            plt.title('gcm 500hPa height '+str(aggregation)+'day variance')
            plt.savefig('puma_'+str(aggregation)+'dayvariance_leadtime'+str(lead_time)+'.pdf')
    



#%% make a longer climate run (1000 years)
# TODO: plot timeline of aggregated data (e.g. mean temperature of the earth, yearly averages)
# to see whether there is a drift
    
var = 'zg500'

plt.figure()
clim_newtork_agg = clim_run.resample(time='360d').mean('time')
timeline = clim_newtork_agg.isel(lev=varname_to_levidx[var]).mean(('lat','lon'))
timeline.plot()
plt.savefig('timeline_network_clim_yearmean_'+var+'.pdf')


# we start the long climate run from 20 randomly sampled initial conditions
for count_init, i_init in enumerate(np.random.randint(0,len(X_test), size=20)):
    
    N_years = 1000
    
    # the number of forecast steps to make depends on the lead_time
    
    N_steps = int(np.round(N_years * 365 / lead_time))
    
    initial_state = X_test[i_init]    
    
    # the long run needs a lot of memory, so we only store the area mean
    nlevs = X_test.shape[-1]
    clim_run_long = np.zeros(shape=[N_steps+1,nlevs], dtype='float32')
    
    clim_run_long = xr.DataArray(data = clim_run_long, coords={'time':np.arange(N_steps+1),
                                                     'lev':X_test.lev},dims=('time','lev')
        )
    
    
    new_state = np.expand_dims(initial_state,0)
    clim_run_long[0] = np.squeeze(new_state.mean(axis=(1,2)))
    for i in range(N_steps):
        print(i,N_steps)
        # we have to add an (empty) time dimenstion
        new_state = model.predict(new_state)
        prediction_areamean = new_state.mean(axis=(1,2))
        clim_run_long[i+1] = np.squeeze(prediction_areamean)
        
    
        
    # 3compute yearly mean    
    # because the timeseries is so long, the pandas datetime things fail. therefore,
    # we do it different. First compute a rolling 360 day mean, and then extract every 360th
    # timestep. for this we need to divide 360 by the lead time
    

    agg_years = 5
    freq = int(365*agg_years/lead_time)
    clim_network_agg = clim_run_long.rolling(time=freq).mean()[::freq]
        
    sns.set_style('ticks')
    sns.set_context('talk',rc={"lines.linewidth": 3})
    colorblind=[ "#0072B2","#D55E00","#009E73", 
                    "#CC79A7", "#F0E442", "#56B4E9"] ## colorblind paletter from seaborn with switched order
    sns.set_palette(colorblind)
    plt.rcParams["errorbar.capsize"] = 5
       
    # plot yearly mean for different variables in one plot
    plt.figure(figsize=(8,4)) 
    for var in ('zg500',):#,'ua300','ta800'):
        timeline = clim_network_agg.isel(lev=varname_to_levidx[var])
        plt.plot(np.arange(len(timeline))*agg_years,timeline,label=var)
    plt.legend()    
    plt.xlabel('time [years]')
    sns.despine()
    plt.savefig('timeline_network_clim_yearmean_zg500_init_leadtime'+str(lead_time)+'_'+str(count_init)+'.pdf')    
    
    
    plt.figure(figsize=(8,4)) 
    for ivar in range(clim_network_agg.shape[1]):
        
        plt.plot(np.arange(len(timeline))*agg_years,clim_network_agg[:,ivar])
    plt.legend()    
    plt.xlabel('time [years]')
    sns.despine()
    plt.savefig('timeline_network_clim_yearmean_allvar_init_leadtime'+str(lead_time)+'_'+str(count_init)+'.pdf')    
    
    
    
    del clim_run_long