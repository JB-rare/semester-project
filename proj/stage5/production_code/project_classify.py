"""

Project_classify.py
CS 487, Spring 2020
Semester project
28 Apr 2020
Author: Bill Hanson
@billhnm
version 2.0

This is the main program for reading in the trained vectors,
    running them through the appropriate classification module,
    writing predictions to disk,
    and reporting results.

"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
# add imports as needed, but they should be in your programs

#
## Define needed functions
#


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

algos = ['lr', 'mnb', 'mlp', 'rf', 'svm', 'vote']

algo = algo.lower()

# test for correct algorithm input (ignore capitalization issues)
if algo not in algos:
    print('Your algorithm,', algo, 'is not recognized.')
    print('Please make sure you specify one of the following algorithms:',
    'LR, MNB, MLP, RF, SVM, VOTE.')
    sys.exit('Incorrect algorithm specified, terminating.')
#%%
#
## Read the appropriate datafiles

if algo != 'vote':
    filename_X = algo + '_vectors.npz'
    in_X = np.load(filename_X, allow_pickle=True)
    # get train and test vectors
    X_train = in_X['train_vec']
    X_test = in_X['test_vec']
elif algo == 'vote':
    # if we're voting, read in all the various predictions
    in_lr = np.load('lr_preds', allow_pickle=True)
    y_train_lr = in_lr['train_preds']
    y_test_lr = in_lr['test_preds']
    # mlp
    in_mlp = np.load('mlp_preds', allow_pickle=True)
    y_train_mlp = in_mlp['train_preds']
    y_test_mlp = in_mlp['test_preds']
    # mnb
    in_mnb = np.load('mnb_preds', allow_pickle=True)
    y_train_mnb = in_mnb['train_preds']
    y_test_mnb = in_mnb['test_preds']
    # rf
    in_rf = np.load('rf_preds', allow_pickle=True)
    y_train_rf = in_rf['train_preds']
    y_test_rf = in_rf['test_preds']
    # svm
    in_svm = np.load('svm_preds', allow_pickle=True)
    y_train_svm = in_svm['train_preds']
    y_test_svm = in_svm['test_preds']
else:
    print('something wrong')

# read in y (target) values

filename_y = 'IMDB_5k_y.npz'
# filename_y = 'IMDB_y.npz'
in_y = np.load(filename_y, allow_pickle=True)
y_train = in_y['y_train']
y_test = in_y['y_test']

#%%
#
## Figure out which preproc tool to run based on 'algo'
#

# assign hyperparameter variables based on algorithm
# comment out for testing

if algo == 'lr':
    # int_param = int(params[0])
    max_iter_ = 7800
    random_state_ = 42
    # str_param = params[1]
    solver_ = 'liblinear'
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
    svm_kernel = linear   # linear or rbf
    svm_gamma = 0.1   # some float
    pass
elif algo == 'vote':
    # int_param = int(params[0])
    # str_param = params[1]
    # flt_param = float(parmam[2])
    pass

# Set parameters for testing purposes
#
# comment out for turn-in
'''
algo = 'lr'


'''

## select module to call based on algo
#
#   Call program module and function
#   Pass X_train and X_test
#   Return predictions for train and test

if algo == 'lr':
    # call the external module and function
    from project_LR import lr_preproc
    lr_train_vec, lr_test_vec = lr_preproc(X_train, X_test)
    from project_LR import lr_classify
    lr_train_preds, lr_test_preds = lr_classify(X_train, y_train, lr_train_vec, lr_test_vec, max_iter_, solver_, random_state_) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('lr_preds', train_preds = lr_train_preds, test_preds = lr_test_preds)
elif algo == 'mnb':
    # call the external module and function
    from proj_mnb import mnb_classify
    mnb_train_preds, mnb_test_preds = mnb_classify(X_train, X_test) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('mnb_preds', train_preds = mnb_train_preds, test_preds = mnb_test_preds)
elif algo == 'mlp':
    # call the external module and function
    from proj_mlp import mlp_classify
    mlp_train_preds, mlp_test_preds = mlp_classify(X_train, X_test) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('mlp_preds', train_preds = mlp_train_preds, test_preds = mlp_test_preds)
elif algo == 'rf':
    # call the external module and function
    from proj_rf import rf_classify
    rf_train_preds, rf_test_preds = rf_classify(X_train, X_test) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('rf_preds', train_preds = rf_train_preds, test_preds = rf_test_preds)
elif algo == 'svm':
    # call the external module and function
    from project_svm import svm_classify
    svm_train_preds, svm_test_preds = svm_classify(X_train, X_test, y_train, y_test, svm_kernel, svm_gamma) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('svm_preds', train_preds = svm_train_preds, test_preds = svm_test_preds)
elif algo == 'vote':
    ## Add the arrays together
    y_train_add = y_train_lr + y_train_mlp + y_train_mnb + y_train_rf + y_train_svm
    y_test_add = y_test_lr + y_test_mlp + y_test_mnb + y_test_rf + y_test_svm
    # if result is positive, set prediction to 1, else -1
    train_preds = np.where(y_train_add > 0, 1, -1)
    test_preds = np.where(y_test_add > 0, 1, -1)
else:
    print('Nothing done: something went wrong')# error out
    print('Check your premises')

# if we voted, then go to report the results, otherwise say goodbye and exit
if algo != 'vote':
    print('All done with classification run!')
    sys.exit('Classification run complete, exiting.')

#%%
#
## Report the results

print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
'''Program end'''
