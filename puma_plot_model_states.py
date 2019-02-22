#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:25:01 2018


reads in postprocessed PUMA output, and reorders the data
so that it is more convenient for the training

@author: sebastian
"""
import numpy as np
import pandas as pd
import xarray as xr

from pylab import plt

plt.rcParams['savefig.bbox']='tight'


# we pick out one aribtary year of the simulation (but not from the beginning, which might
# suffer from spinup-issues)
ifile = '/climstorage/sebastian/plasim/testrun/testrun.500_plevel.nc'

    
print('open inputdata', ifile)
data = xr.open_dataset(ifile, chunks={'time':1},decode_times=False)
    

    
tidx = 100
lev=500
var = 'zg'

x = data[var].isel(time=tidx).sel(lev=lev)

plt.figure(figsize=(7,3))
x.plot.contourf(levels=10, cmap=plt.cm.plasma_r)

plt.savefig('puma_'+var+'_lev'+str(lev)+'_'+str(tidx)+'.pdf')



lev = 300
var = 'ua'
x = data[var].isel(time=tidx).sel(lev=lev)

plt.figure(figsize=(7,3))
x.plot.contourf( cmap=plt.cm.RdBu_r, levels=[-60,-50,-40,-30,-20,-10,10,20,30,40,50,60])

plt.savefig('puma_'+var+'_lev'+str(lev)+'_'+str(tidx)+'.pdf')