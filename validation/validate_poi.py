#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys="../tools/python2_lesson13_keys.pkl")
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
clf_score = clf.score(features_test,labels_test)
print(f"Classifier accuracy after applying split conditions: {clf_score}")

count_pre_poi = len([e for e in clf.predict(features_test) if e == 1.0])
print(f"no. of identified poi's in test set: {count_pre_poi}")
count_test_all = len(labels_test)
print(f"no. of all people in test set: {count_test_all}")
count_test_poi = len([e for e in labels_test if e == 1.0])
# print(count_test_poi)
acc_zero_poi = 1-(count_test_poi/count_test_all)
print(f"Accuracy percentage if identifier predicts (not poi) only: {acc_zero_poi}")

from sklearn.metrics import precision_score, recall_score
pre_score = precision_score(labels_test,clf.predict(features_test))
print(f"Precision score for test set and predicted set: {pre_score}")
rcl_score = recall_score(labels_test,clf.predict(features_test))
print(f"Recall score for test set and predicted set: {rcl_score}")
