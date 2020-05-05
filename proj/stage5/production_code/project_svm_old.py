#!/usr/bin/env python
# -*- coding: utf-8
"""

NMSU - CS519 - Spring 2020

Written by:  Eloy Macha
Date written:

Purpose:

Input: <main>.py -c CVAR -i IVAR -t TVAR  [-o]

<Some Description.....>

  -c CVAR, --cvar CVAR
                        cvar: choices
  -i IVAR, --ivar IVAR
                        ivar: choices
  -t TVAR, --tvar TVAR
                        tvar: choices
  -o, --ovr             Flag Optional


Output:

@author: Frank
"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#
## Define needed functions
#

# init stemmer
porter_stemmer=PorterStemmer()

def my_preprocessor(text):
    # pre-process to remove special chars, lower-case, normalize and stem
    text=text.lower()
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words

    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)


def clean_df(frame):

    # assuming frame is single column

    string_array = frame.to_numpy()
    new_string_array = []

    for entry in range(0,len(string_array)):

        old_string = string_array[entry]

        # Remove all the special characters
        clean_string = re.sub(r'\W', ' ', str(old_string))

        # remove all single characters
        clean_string = re.sub(r'\s+[a-zA-Z]\s+', ' ', clean_string)

        # Remove single characters from the start
        clean_string = re.sub(r'\^[a-zA-Z]\s+', ' ', clean_string)

        # Substituting multiple spaces with single space
        clean_string = re.sub(r'\s+', ' ', clean_string, flags=re.I)

        # Removing prefixed 'b'
        clean_string = re.sub(r'^b\s+', '', clean_string)

        # Converting to Lowercase
        new_string = X_processed_entry.lower()

        new_string_array.append(new_string)

    return new_string_array


#%%
#
## functions called from project_preproc or project_classify
#
def svm_preproc(X_train, X_test):

    # all preprocessing code here

    X_train_clean = clean_df(X_train)
    X_test_clearn = clean_df(X_test)

    nltk.download('stopwords')

    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    X_train_vec = vectorizer.fit_transform(X_train_clean).toarray()
    X_test_vec = vectorize.fit_transform(X_test_clean).toarray()

    return X_train_vec, X_test_vec

def svm_classify(X_train, X_test, y_train, y_test, tun_kernel, tun_gamma ):
    # all classification code here

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    if tun_kernel == 'linear':
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1)
    else:
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1, gamma=tun_gamma)

    # run .fit - capture time
    start_time=time.time()
    svm.fit(X_train, y_train)
    run_time=time.time() - start_time

    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    misclass = (y_test_pred != y_test).sum()

    svm_train = accuracy_score(y_train, y_train_pred)
    svm_test = accuracy_score(y_test, y_test_pred)

    return svm_train, svm_test
