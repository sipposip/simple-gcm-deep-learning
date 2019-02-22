
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

from pylab import plt


plt.rcParams['savefig.bbox'] = 'tight'

from dask.diagnostics import ProgressBar
ProgressBar().register()

def corr_over_time(x,y):
    '''
        point-by point correlation of two spatial arrays x and y
        (the correlation of x[all_times,lati,loni] and y[all_times,lati,loni])
        Can be used to make correlation maps of two quantities
        
    '''
    if x.shape != y.shape:
        raise ValueError('x and y must have exactly the same dimension!')    
  
    # the numpy corrcoref function works only for one point at a time, 
    # and looping over the field is very slow. therefore we do the computation
    # on a lower level (only wieh mean and std functions)
    mx = x.mean('time')
    my = y.mean('time')
    xm, ym = x-mx, y-my
    r_num = (xm*ym).mean('time')
    r_den = xm.std('time') * ym.std('time')
    
    r = r_num / r_den
    return r

def weighted_areamean(ds):
    '''area mean weighted by cos of latitude'''
    weights = xr.ufuncs.cos(np.deg2rad(ds.lat))
    # this is 1d array. Xarray broadcasts it automatically.
    # however, we compute the normalization term explicitely.
    # the normalization term is the sum of the weights over all gridpoints.
    # because the weight is not lon dependent, is is simply the sum along one 
    # meridian times the number of lon points
    norm = np.sum(weights) * len(ds.lon)
    
    # this means we have only lat and lon
    amean = (ds*weights).sum(('lat','lon')) / norm

    return amean


# the input file was produced with "puma_CNN_preprocess_inputdata_v2.py"
# it is the full puma output, put into one file, reordered coordinates to
# ('time','lat','lon','lev'), and normalized every level to a mean of 0 and
# a std of 1

puma_ifile='/climstorage/sebastian/plasim/testrun/testrun.reordered.normalized.merged.nc'
ml_prediction_dir='/climstorage/sebastian/simiple_gcm_machinelearning/tuned/'


# the level and variable dimension is the same, it is ua100hPa, ua200hPa,....ua10000hPa, va100hPa....]
varnames = ['ua','va','ta','zg']
keys = [ varname+str(lev) for varname in varnames for lev in range (100,1001,100)]
varname_to_levidx = { key:levidx for key,levidx in zip(keys,range(len(keys)))  }

# dictionary for conerting varname and level to "lev" in the processed data (
# in the processed data, all variables are stacked along lev)




print('open inputdata')

puma_data = xr.open_dataarray(puma_ifile, chunks={'time':1})

# note that the time-variable in the input file is confusing: it contains the
# day of the year of the simulation, thus it is repeating all the time 
# (it loops from 1 to 360 and then jumps back to one)


## define the data

## first define spinup years, this are the first N years
## that will be discarded from the data

spinup_years = 30
N_spinup = 365*spinup_years
data = puma_data.isel(time=slice(N_spinup,None))


# for the tuning, the data was split up into train:dev
# this was not the best choice, as now it gets complicated when increasing the
# amount of training data. therefore, we dont use the training data used in the
# tuning.  we discard it, and then use exactly the same development data, then use the next
# years as test data, and finally the rest as training data

## here, we in fact only need the trianing data (to compute  a climatoogical prediction)

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




def acc_score(x,y):
    '''timestepwise anomaly correlation coefficient, averaged over time
        (simple version without seasonal climatoloty)'''
    assert(x.shape==y.shape)
    return np.mean([np.corrcoef(x[i].flatten(),y[i].flatten())[0,1] for i in range(len(x))])
    
    

rmse_per_leadtime = []
corr_per_leadtime = []
acc_per_leadtime  = []
clim_rmse_per_leadtime = []
persistance_rmse_per_leadtime = []
persistance_corr_per_leadtime = []
persistance_acc_per_leadtime = []

lead_times = np.arange(1,14+1)
for lead_time in lead_times:

    # open predictions and true vales for this lead time
    
    preds = xr.open_dataarray(ml_prediction_dir+'/predictions_tuned_leadtime'+str(lead_time)+'params_36500_10_'+str(lead_time)+'.nc')
    truth = xr.open_dataarray(ml_prediction_dir+'/truevalues_tuned_leadtime'+str(lead_time)+'params_36500_10_'+str(lead_time)+'.nc')
    
    # compute RMSE over time
    
    rmse_timmean = np.sqrt(((preds - truth)**2).mean('time'))
    
    rmse_timmean_areamean = weighted_areamean(rmse_timmean)
    rmse_per_leadtime.append(rmse_timmean_areamean)
    
    # correlation over time (per gridpoint)
        
    corr_timmean = corr_over_time(preds,truth)
    corr_per_leadtime.append(weighted_areamean(corr_timmean))
    # mapplots of RMSE for selected variables
    
    # anomaly correlation, this does not work with xr.Datarrays, because it needs
    # the flatten attribuite, therefore convert to numpy first
    acc = acc_score(preds.values,truth.values)
    acc_per_leadtime.append(acc)
    
        cmap=plt.cm.gist_heat_r
        levels=np.arange(0,0.241,0.01)
        for var in ['zg500','ua300']:
            plt.figure(figsize=(7,3))
            rmse_timmean.isel(lev=varname_to_levidx[var]).plot.contourf(cmap=cmap, levels=levels)
            plt.title('network forecast rmse')
            plt.savefig('puma_rmse_map'+var+'_'+str(lead_time)+'.pdf')
        
        
        
    
    # mapplots of correlation for selected variables
    
    
    for var in ['zg500','ua300']:
        plt.figure(figsize=(7,3))
        corr_timmean.isel(lev=varname_to_levidx[var]).plot.contourf(levels=np.arange(-1,1.01,0.1))
        plt.title('network forecast correlation')
        plt.savefig('puma_corr_map_'+var+'_'+str(lead_time)+'.pdf')
    
    
    X_train,y_train, X_dev, y_dev, X_test, y_test = prepare_data(lead_time)
    
    
    clim = X_train.mean('time').load()
    
    # compute the error of a climatological forecast on the test data
    
    X_test.load()
    
    clim_rmse_timmean = np.sqrt(((X_test - clim)**2).mean('time'))
        
    clim_rmse_per_leadtime.append(weighted_areamean(clim_rmse_timmean))

    for var in ['zg500','ua300']:
        plt.figure(figsize=(7,3))
        clim_rmse_timmean.isel(lev=varname_to_levidx[var]).plot.contourf(cmap=cmap, levels=levels)
        plt.title('climatology forecast rmse')
        plt.savefig('clim_rmse_map_'+var+'.pdf')
        
        
    ## presistence
        
    ## here, we assume use X_test as prediction
    
    # we need to update the time in X_test for the prediction, for this we copy
    # it and use the y_test dimensions
    y_test.load()
    
    persistance_prediction = xr.DataArray(data=X_test.data,coords=y_test.coords,dims=y_test.dims)
    
    
    persistence_rmse_timmean = np.sqrt(((y_test - persistance_prediction)**2).mean('time'))  
    
    persistance_rmse_per_leadtime.append(weighted_areamean(persistence_rmse_timmean))
    
    
    persistance_corr_timmean = corr_over_time(y_test,persistance_prediction)
    
    persistance_corr_per_leadtime.append(weighted_areamean(persistance_corr_timmean))
    
    persistance_acc = acc_score(y_test.values,persistance_prediction.values)
    persistance_acc_per_leadtime.append(persistance_acc)
    # (for acc, we cannot make mapplots because it aggreagtes over spatial dims)
    
    for var in ['zg500','ua300']:
        plt.figure(figsize=(7,3))
        persistence_rmse_timmean.isel(lev=varname_to_levidx[var]).plot.contourf(cmap=cmap, levels=levels)
        plt.title('persistence forecast rmse')
        plt.savefig('persitance_rmse_map_'+var+'.pdf')
    
    for var in ['zg500','ua300']:
        plt.figure(figsize=(7,3))
        persistance_corr_timmean.isel(lev=varname_to_levidx[var]).plot.contourf(levels=np.arange(-1,1.01,0.1))
        plt.title('persistence forecast correlation')
        plt.savefig('persitance_corr_map_'+var+'_'+str(lead_time)+'.pdf')



sns.set_style('ticks')
sns.set_context('talk',rc={"lines.linewidth": 3})
colorblind=[ "#0072B2","#D55E00","#009E73", 
                "#CC79A7", "#F0E442", "#56B4E9"] ## colorblind paletter from seaborn with switched order
sns.set_palette(colorblind)
plt.rcParams["errorbar.capsize"] = 5

for var in ['zg500','ua300']:
    ## plot lead time vs scores
    plt.figure(figsize=(8,4))        
    plt.plot(lead_times, [e[varname_to_levidx[var]].values for e in rmse_per_leadtime],
        label='network')  
    plt.plot(lead_times, [e[varname_to_levidx[var]].values for e in clim_rmse_per_leadtime],
             label='climatology')      
    plt.plot(lead_times, [e[varname_to_levidx[var]].values for e in persistance_rmse_per_leadtime],
             label='persistence') 
    sns.despine()
    plt.ylabel('rmse [normalized data]')
    plt.xlabel('lead time [days]')
    plt.legend()
    plt.savefig('lead_time_vs_skill_'+var+'.pdf')
    
    plt.figure(figsize=(8,4))        
    plt.plot(lead_times, [e[varname_to_levidx[var]].values for e in corr_per_leadtime],
        label='network')     
    plt.plot(lead_times, [e[varname_to_levidx[var]].values for e in persistance_corr_per_leadtime],
             label='persistence') 
    sns.despine()
    plt.ylabel('correlation')
    plt.xlabel('lead time [days]')
    plt.legend()
    plt.savefig('lead_time_vs_corr_'+var+'.pdf')
    
    plt.figure(figsize=(8,4))        
    plt.plot(lead_times, acc_per_leadtime,
        label='network')     
    plt.plot(lead_times, persistance_acc_per_leadtime,
             label='persistence') 
    sns.despine()
    plt.ylabel('anomaly correlation')
    plt.xlabel('lead time [days]')
    plt.legend()
    plt.savefig('lead_time_vs_acc_'+var+'.pdf')