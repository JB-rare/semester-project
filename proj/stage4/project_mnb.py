"""

Project_mnd.py
CS 487, Spring 2020
Semester project
6 Mar 2020
Author: Bill Hanson
@billhnm
version 1.5

This program reads in the IMDB training dataset and performs basic 
    multinomial naive bayes classification

"""
#%%
# # Import needed packages

import os
import sys
import argparse 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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

#%%
# 
## Read the IMDB datafile
# filename = 'IMDBtrain.csv'
filename = 'imdb_1250.data'

# df_raw = pd.read_csv(filename, header = None, sep = None, engine = 'python') # read data file
df_raw = pd.read_csv(filename, header = None) # read data file

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx


## Normal pre-proc stuff
#

y, X = split_data(df_raw)

X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=1, stratify=y) 

#%%
#
## turn to df
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)

#%%
#
## Vectorize the data

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(min_df=2)
train_vec = vectorizer.fit_transform(df_X_train[0])
test_vec = vectorizer.transform(df_X_test[0])

#%%
#
## run through MNB

mnb = MultinomialNB()
mnb.fit(train_vec, y_train)
train_preds = mnb.predict(train_vec)
test_preds = mnb.predict(test_vec)

print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
# Program end