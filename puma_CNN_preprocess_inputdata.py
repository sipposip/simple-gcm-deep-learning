#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:25:01 2018


reads in postprocessed PUMA output, and reorders the data
so that it is more convenient for the training

@author: sebastian
"""
import matplotlib
matplotlib.use('agg')
import numpy as np
import pandas as pd
import xarray as xr

import tensorflow as tf


from dask.diagnostics import ProgressBar
ProgressBar().register()


#ifile = '/proj/bolinc/users/x_sebsc/simple_gcm_ml/puma/testrun/testrun_plevel_merged.nc'
#ifile='/home/s/sebsc/pfs/data/simple_gcm_machinelearning/testrun/testrun_plevel_merged.nc'
ifiles=['/climstorage/sebastian/plasim/testrun/testrun.'+str(year).zfill(3)+'_plevel.nc' for year in range(1,1000)]

for ifile in ifiles:
    
    print('open inputdata', ifile)
    data = xr.open_dataset(ifile, chunks={'time':1},decode_times=False)
    
    # we have different variables, each dimension (time,lev,lat,lon)
    # we want to stack all variables, so that we have dimension (time,lat,lon,channel)
    # where channel is lev1,lev2... of variable 1, lev1,lev2,... of variable 2 and son on
    
 
    
    
    print(' stack data')
    varnames = ['ua','va','ta','zg']
    # stack along level dimension
    stacked = xr.concat((data[varname] for varname in varnames), dim='lev')
    
    
    print('reorder data')
    # now reorder so that lev is last dimension
    x = stacked.transpose('time','lat','lon','lev')
    
    
    
    ofile = ifile+'.reordered.nc'
    #
    x.to_netcdf(ofile)
    
    data.close()
    x.close()
    
    
    
## normalize
ifiles=['/climstorage/sebastian/plasim/testrun/testrun.'+str(year).zfill(3)+'_plevel.nc.reordered.nc' for year in range(1,1000)]


data = xr.open_mfdataset(ifiles, decode_times=False, concat_dim='time')

mm = data.mean(('time','lat','lon'))
std = data.std(('time','lat','lon'))

mm.load()
std.load()

normalized = (data - mm) / std


normalized.to_netcdf('/climstorage/sebastian/plasim/testrun/testrun.reordered.normalized.merged.nc')


    
    
    
    
