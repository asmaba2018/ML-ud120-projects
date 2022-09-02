#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(C=10000., kernel="rbf")
C_list = [10., 100., 1000., 10000.]


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
# t0 = time()
clf.fit(features_train, labels_train)
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# clf.predict(features_test)
# print("Training Time:", round(time()-t0, 3), "s")

# for C_value in C_list:
#     clf = svm.SVC(C=C_value, kernel="rbf")
#     clf.fit(features_train, labels_train)
#     accuracy = clf.score(features_test, labels_test)
#     print(accuracy)

accuracy = clf.score(features_test, labels_test)
print(accuracy)
#########################################################
