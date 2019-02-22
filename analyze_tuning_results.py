#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:54:48 2018

@author: sebastian
"""

import pickle
import json

import numpy as np

res = pickle.load(open('tuning_result10950_4_32_5_2_0.pkl','rb'))


# get the final validatino loss from every run

final_val_losses = [e['hist']['val_loss'][-1] for e in res]
# conver to array
final_val_losses = np.array(final_val_losses)
# get the index of the smallest

idx_smallest = np.argmin(final_val_losses)


res_best = res[idx_smallest]

print(res_best)

with open('tuning_best_result.txt','w') as f :
    f.write(str(res_best))