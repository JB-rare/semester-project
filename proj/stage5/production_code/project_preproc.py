"""

Project_preproc.py
CS 487, Spring 2020
Semester project
28 Apr 2020
Author: Bill Hanson
@billhnm
version 2.0

This is the main program for reading in the IMDB files, running appropriate
    preprocessing, and then writes the train/test vectors to disk.  

"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse 
import pandas as pd
import numpy as np
# add imports as needed, but they should be in your programs

#
## Define needed functions
#

def split_data(df):
    # splits last column (tgt - classification labels) from data into a new df
    nf = df.shape[1] - 1
    tgt = df.iloc[:, -1].values
    feats = df.iloc[:, 0:nf].values
    return tgt, feats

def join_data(df, y):
    # adds classification labels back to the last column
    df_new = df.copy()
    df_new = pd.DataFrame(df_new) 
    col = df_new.shape[1]
    df_new[col] = y
    return df_new

#
#%%
## Inputs algorithm, filename and up to four parameters

# initialize variables
random_state = 42
params = [None] * 4

#
# read the algorithm and parameters specified in the command line (comment out for development)
# NOTE: assumes files are the two full IMDB sets  

# NOTE (Question): do we need to add an argument for a separate stopwords file? 
parser = argparse.ArgumentParser()
parser.add_argument('algo')
parser.add_argument('params', default=None, nargs='*')

args = parser.parse_args()

algo = args.algo
params = args.params

algos = ['lr', 'mnb', 'mlp', 'rf', 'svm', 'all']

algo = algo.lower()

# test for correct algorithm input (ignore capitalization issues)
if algo not in algos:
    print('Your algorithm,', algo, 'is not recognized.')
    print('Please make sure you specify one of the following algorithms:',  
    'LR, MNB, MLP, RF, SVM, ALL.')
    sys.exit('Incorrect algorithm specified, terminating.')
#%%
# 
## Read the IMDB datafile

# trainfile = 'IMDBtrain.csv'
# testfile = 'IMDBtest.csv'
trainfile = 'IMDB_5k_train.data'
testfile = 'IMDB_5k_test.data'

train_raw = pd.read_csv(trainfile, header = None) # read data file
test_raw = pd.read_csv(testfile, header = None)

## Read stop_words files - depending on preprocessing
#
stopfile = 'imdb_stop_words.data' 
# imdb_stop_words = pd.read_csv(stopfile, header = None)
# create a set from the df
# imdb_stop_words = set(imdb_stop_words[0])

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx

## split test/train
#
y_train, X_train = split_data(train_raw)
y_test, X_test = split_data(test_raw)

## turn to df
#
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)

#%%
#
## Figure out which preproc tool to run based on 'algo'
#

# assign hyperparameter variables based on algorithm
# comment out for testing

if algo == 'lr':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass
elif algo == 'mnb':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass
elif algo == 'mlp':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass
elif algo == 'rf':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass
elif algo == 'svm':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass
elif algo == 'all':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass

# Set parameters for testing purposes
# ['k-means', 'sp-hierarchical', 'sk-hierarchical', 'dbscan']
# comment out for turn-in
'''
algo = 'lr'
stopwords = imdb_stop_words

'''

## select module to call based on algo
#
#   Call program module and function
#   Pass X_train and X_test
#   Return preprocessed vectors train and test  

if algo == 'lr':
    # call the external module and function
    from proj_lr import lr_preproc
    lr_train_vec, lr_test_vec = lr_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('lr_vectors', train_vec = lr_train_vec, test_vec = lr_test_vec)
elif algo == 'mnb':
    # call the external module and function
    from proj_mnb import mnb_preproc
    mnb_train_vec, mnb_test_vec = mnb_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('mnb_vectors', train_vec = mnb_train_vec, test_vec = mnb_test_vec)
elif algo == 'mlp':
    # call the external module and function
    from proj_mlp import mlp_preproc
    mlp_train_vec, mlp_test_vec = mlp_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('mlp_vectors', train_vec = mlp_train_vec, test_vec = mlp_test_vec)
elif algo == 'rf':
    # call the external module and function
    from proj_rf import rf_preproc
    rf_train_vec, rf_test_vec = rf_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('rf_vectors', train_vec = rf_train_vec, test_vec = rf_test_vec)
elif algo == 'svm':
    # call the external module and function
    from proj_svm import svm_preproc
    svm_train_vec, svm_test_vec = svm_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('svm_vectors', train_vec = svm_train_vec, test_vec = svm_test_vec)
elif algo == 'all':
    # do something
    pass
else:
    print('Nothing done: something went wrong')# error out
    print('Check your premises')

print('All done!')
#
#%%
'''Program end'''
