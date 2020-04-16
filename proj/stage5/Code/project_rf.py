"""

Project_rf.py
CS 487, Spring 2020
Semester project
16 Apr 2020
Author: Bill Hanson
@billhnm
version 2.0

This program reads in the IMDB training dataset and performs basic 
    random forest classification

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

#%%

# 
## Read the IMDB datafile
# filename = 'IMDBtrain.csv'
filename = 'imdb_5k_train.data'

## Read stop_words files - 2 versions depending on preprocessing
imdb_stop_words = pd.read_csv('imdb_stop_words_stem.data', header = None)
imdb_stop_words_nostem = pd.read_csv('imdb_stop_words.data', header = None)

# df_raw = pd.read_csv(filename, header = None, sep = None, engine = 'python') # read data file
df_raw = pd.read_csv(filename, header = None) # read data file
df_raw.columns=['review', 'target']

# get the nltk stopword file too
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

nltk.download('vader_lexicon')

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx

#%%
#
## clean the text
df_raw["clean_review"] = df_raw['review'].apply(lambda x: clean_text(x))

#%%
#
## add sentiment data
sid = SentimentIntensityAnalyzer()
df_raw["sentiments"] = df_raw['review'].apply(lambda x: sid.polarity_scores(x))
df_raw = pd.concat([df_raw.drop(['sentiments'], axis=1), 
        df_raw['sentiments'].apply(pd.Series)], axis=1)

#%%
#
## Write current file to disk

#
df_raw.to_csv('df_raw_inter.data', header = True, index = False)

#%%
#
## read the file in if needed
df_raw = pd.read_csv('df_raw_inter.data', header = 0)

#%%
# add number of words column
df_raw["nb_words"] = df_raw["review"].apply(lambda x: len(x.split(" ")))

# and make a copy to work with
df_prep = df_raw
#%%
#
## create doc2vec columns

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df_prep['clean_review'].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=4, window=2, min_count=2, workers=4)

# transform each document into a vector data
doc2vec_df = df_prep['clean_review'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([df_prep, doc2vec_df], axis=1)

#%%
#
## add tf-idf vectors
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,2))
tfidf_result = vectorizer.fit_transform(df_prep['clean_review']).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = vectorizer.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df_prep.index
review_vec_df = pd.concat([df_prep, tfidf_df], axis=1)             

#%%
#
## write reviews_df to file

review_vec_df.to_csv('reviews_vectorized.data', header = True, index = False)

#%%
#
## plot sentiment distribution for positive and negative reviews

import seaborn as sns

for x in [-1, 1]:
    subset = review_vec_df[review_vec_df['target'] == x]
    
    # Draw the density plot
    if x == 1:
        label = "Good reviews"
    else:
        label = "Bad reviews"
    sns.distplot(subset['compound'], hist = False, label = label)


#%%
## Normal pre-proc stuff
#
# feature selection
label = 'target'
ignore_cols = [label, 'review', 'clean_review']
features = [c for c in review_vec_df.columns if c not in ignore_cols]

# split the data into train and test
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(review_vec_df[features], 
                review_vec_df[label], test_size = 0.30, random_state = 42)

#%%
#
## train a random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)

#%%
#
## ROC curve

y_pred = [x[1] for x in rf.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%
#
## print results and scores

train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)

print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
#
X = df_raw[['review', 'clean_review', 'neg', 'neu', 'pos', 'compound', 'nb_words']].values
y = df_raw[['target']].values

#
X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=1, stratify=y) 

#%%
#
'''
## turn to df
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)
'''

#%%
#
### fancier version with stop words, n-grams, etc

## Vectorize the data

# (ngram_range=(1,2), stop_words=imdb_stop_words, preprocessor=my_preprocessor)
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2),
                preprocessor=my_preprocessor, stop_words=imdb_stop_words)
train_vec = vectorizer.fit_transform(df_X_train[0])
test_vec = vectorizer.transform(df_X_test[0])

#%%
## run through SVM

from sklearn import svm

svc = svm.SVC()
svc.fit(X_train, y_train)
train_preds = svc.predict(X_train)
test_preds = svc.predict(X_test)

print('SVM - pass 2')
print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
#
## show roc_auc curve
# predict probabilities
mnb_probs = mnb.predict_proba(test_vec)
# separate for positive 
mnb_probs_p = mnb_probs[:, 1]
# calculate scores
mnb_p_auc = roc_auc_score(y_test, mnb_probs_p)
# calculate the roc curves
p_fpr, p_tpr, _ = roc_curve(y_test, mnb_probs_p)
# plot the roc curve for the model
plt.plot(p_fpr, p_tpr, linestyle='--', label='Positive Review')
# plt.plot(n_fpr, n_tpr, marker='.', label='Negative Review')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# add the auc score in a text box
auc_score = roc_auc_score(y_test, test_preds)
plt.text(0.65, 0.2, ("AUC Score:", round(auc_score,3)))
# show the plot
plt.show()
#%%





#%%
#
## create custom stop words with min_df and max_df 
#   Ignores words occuring in very few reviews (lots of terms in neg 
#       reviews that only occur a couple of times, so wont do this)
#   Ignores words occuring in majority (>50% for now) of reviews

# Create stop word list for terms occuring in >50% of reviews
cv = CountVectorizer(max_df=.5, preprocessor=my_preprocessor)
# cv = CountVectorizer(max_df=.5) #create non-stemmed version
count_vector=cv.fit_transform(df_X_train[0])
imdb_stop_words  = cv.stop_words_

#%%
#
## write stop words to data
#df_stop_words = pd.DataFrame(data=imdb_stop_words)
#df_stop_words.to_csv('imdb_stop_words.data', header=False, index=False)

#%%

#
#%%
# Program End