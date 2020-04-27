## Status

## Heading for the Finish!

Tested voting classifier on results from RF, MNB, and MLP (All had test accuracy = 0.85)
* simple majority vote on the 3 classifiers increased accuracy to 0.87
* posted proj_voting.py file to code directory

Random Forest Classifier updated!  Better score!
* Test accuracy / f1 / ROC_AUC = improved from 0.83 to 0.85
* revised proj_rf.py file posted to /stage5/code directory
* Revised ROC curve posted to /charts directory -- area under curve = 0.92

Stage 4 files copied to stage 5 directory

Final Stage 4 report posted: Project Report for Stage 4-v3.docx

Files rearranged into sub-directories for ease of review:
* Datasets: 1250 and 5k IMDB subsets, IMDB stop words, dataset description, full dataset URL
* Code: python code for preprocessing, existing classifiers (Logistic Regression, Multinomial Naive Bayes, Multi-layer Perceptron, SVM), etc.
* Charts_results: Initial charts, result summaries, etc. 
* Batch_instructions_files: batch and readme files for running code

Initial SVM classifier complete
* Test accuracy/f1: 0.84/0.84
* IDFSVM/main.py file posted with associated modules in /modules subdirectory

Initial Logistic Regression classifier complete
* Test accuracy/f1: 0.866/0.866
* project_LR.py file posted

Initial Multi-Layer Perceptron Neural Net Model Results based on 5k dataset
* Test Accuracy / f1 / ROC_AUC: 0.86 / 0.86 / 0.86
* updated project_mlp.py file posted
* Project_mlp_neural_network_results.docx file posted with intermediate and alternate results

Naive Bayes results on 1250 dataset, adding stop words and bigrams
* Test Accuracy / f1 /ROC_AUC: 0.84 / 0.83 / 0.84
* updated project_mnb.py file posted
* Project_mnb_results.docx file posted with intermediate and alternate results
* mnb_roc_curve.png posted
* imdb_stop_words.data file added based on initial analysis

