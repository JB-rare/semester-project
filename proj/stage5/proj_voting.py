"""

proj_voting.py
CS 487, Spring 2020
Semester project
25 Apr 2020
Author: Bill Hanson
@billhnm
version 0.5

This program reads in a set of prediction results and 
uses a 'majority vote' to improve accuracy

"""
#%%
# # Import needed packages

import os
import re
import struct
import sys
import argparse 
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc

#%%
#
## First, create our fake labels.  
#   To make it easy, the first 5,000 = 1 and the 2nd 5,000 = -1

y_pos = np.ones((5000,))
y_neg = np.ones((5000,))*-1

# stack them together

y = np.concatenate((y_pos, y_neg))
# %%
#
## now set up function to assign predictions based on accuracy
#   Create vector of random numbers, then use that to assign prediction

def create_random_preds(y, acc):
    # creates a set of random prediction of given accuracy from
    #   an original set of labels
    y_p = np.copy(y)
    random.seed(datetime.now()) #resets random state before call
    y_r = np.random.random(len(y),)
    for i in range(len(y)):
        if y_r[i] > acc:
            y_p[i] = y_p[i] * -1
    return y_p

#%%
#
## now create 5 different sets of predictions at mid-80s accuracy
y_83 = create_random_preds(y, 0.83)
y_84 = create_random_preds(y, 0.84)
y_85 = create_random_preds(y, 0.85)
y_86 = create_random_preds(y, 0.86)
y_87 = create_random_preds(y, 0.87)

#%%
'''
This is all the function we need to simply use a majority vote 
    approach!
'''
#
## Add all five arrays together
y_add = y_83 + y_84 + y_85 + y_86 + y_87

# if result is positive, set prediction to 1, else -1
y_pred = np.where(y_add > 0, 1, -1)

# calculate accuracy score

print('Accuracy Score %4.3f' %accuracy_score(y, y_pred))

# %%
