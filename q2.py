import numpy as np
import sklearn.linear_model as skl

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler  
test_path='HW2\data\spam.test.dat'
train_path='HW2\data\spam.train.dat'

def do_nothing(train, test):
    return train, test

def do_std(train, test):
    scaler = StandardScaler()
    
    trainx = scaler.fit_transform(train)
    testx = scaler.transform(test)
    return trainx, testx

def do_log(train, test):
    trainx = np.log(train + 0.1)
    testx = np.log(test + 0.1)
    return trainx, testx

def do_bin(train, test):
    trainx = (train > 0).astype(int)
    testx = (test > 0).astype(int)
    return trainx, testx

def eval_nb(trainx, trainy, testx, testy):
    test_prob = np.zeros(testx.shape[0])
    nb_model = GaussianNB()
    
    
    nb_model.fit(trainx, trainy)
    
    train_pred = nb_model.predict(trainx)
    train_prob = nb_model.predict_proba(trainx)[:, 1]  # Probabilities for class 1
    
    train_acc = accuracy_score(trainy, train_pred)
    train_auc = roc_auc_score(trainy, train_prob)
    
    test_pred = nb_model.predict(testx)
    test_prob = nb_model.predict_proba(testx)[:, 1]  # Probabilities for class 1
    
    test_acc = accuracy_score(testy, test_pred)
    test_auc = roc_auc_score(testy, test_prob)
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}

def eval_lr(trainx, trainy, testx, testy):
    test_prob = np.zeros(testx.shape[0])
    # Initialize the Logistic Regression model without regularization
    lr_model = skl.LogisticRegression(max_iter=1000, penalty='none', solver='saga')  # No regularization
    
    # Fit the model on the training data
    lr_model.fit(trainx, trainy)
    
    # Make predictions on the training data
    train_pred = lr_model.predict(trainx)
    train_prob = lr_model.predict_proba(trainx)[:, 1]  # Probabilities for class 1
    
    # Calculate accuracy and AUC for the training set
    train_acc = accuracy_score(trainy, train_pred)
    train_auc = roc_auc_score(trainy, train_prob)
    
    # Make predictions on the test data
    test_pred = lr_model.predict(testx)
    test_prob = lr_model.predict_proba(testx)[:, 1]  # Probabilities for class 1
    
    # Calculate accuracy and AUC for the test set
    test_acc = accuracy_score(testy, test_pred)
    test_auc = roc_auc_score(testy, test_prob)
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}



         
                                
