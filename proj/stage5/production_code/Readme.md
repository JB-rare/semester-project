## Directory for final production code

# Basic structure

# Project_preproc.py
*Read both test and train data files
•	Split target (y) off from each
o	Note: we may want to just do this once and just save the targets as separate files and just read them in when needed
•	Preprocess reviews per individual approaches (one at a time).  Each preprocessing approach is a separate module that takes the test/train review and returns preprocessed X_train and X_test vectors
•	(steps below as required)
o	Cleaning
o	Stemming/Lemmatization
o	Embedding
o	Tokenization
o	Vectorization
•	Write final X_test and X_train vectors to disk in .npz format (numpy.savez) 
o	Note: suggest we use this format – takes much less space than .csv and we can easily read them into the classification section
o	Advantage here is that we can essentially run this once, then just use the created feature vectors as needed

Project_classify
•	Reads in the .npz vectors produced in the preproc section and creates appropriate arrays (numpy.load)
•	Calls individual classifier modules – they would take the train/test arrays and return a vector of predictions
o	Note: I think we need to do this sequentially (e.g., read in Frank’s .npz vectors, runs Frank’s classifier, saves Frank’s predictions, then dump everything except the predictions.  Then step to Jay’s classifier, and so on.)  Otherwise I foresee problems keeping all the feature vectors in memory at the same time. 
•	Combine all predictions via voting and create final prediction plus reports, charts, etc. as needed
