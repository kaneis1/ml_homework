import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as skl
import sklearn.metrics as skm
import pandas as pd
from sklearn.preprocessing import StandardScaler

def _eval_perf(model, x, y):
    preds = model.predict(x)
    probs = model.predict_proba(x)[:, 1]
    acc = skm.accuracy_score(y, preds)
    auc = skm.roc_auc_score(y, probs)
    return acc, auc


def _eval_model(model, trainx, trainy, valx, valy):
    train_acc, train_auc = _eval_perf(model, trainx, trainy)

    val_acc, val_auc = _eval_perf(model, valx, valy)
    
    return {"train-acc": train_acc, "train-auc": train_auc,
            "val-acc": val_acc, "val-auc": val_auc}


def generate_train_val(x, y, valsize):
     
    num_samples = x.shape[0]
  
    indices = np.random.permutation(num_samples)

    split_point = int(num_samples * (1 - valsize))

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_x = x[train_indices]
    train_y = y[train_indices]
    test_x = x[val_indices]
    test_y = y[val_indices]
    return {"train-x": train_x, "train-y": train_y,
            "val-x": test_x, "val-y": test_y}


def generate_kfold(x, y, k):
    fold_assignments=np.zeros(x.shape[0])
    indices = np.random.permutation(x.shape[0])
    for i,indice in enumerate(indices):
        fold_assignments[i] = indice % k
    return fold_assignments


def eval_holdout(x, y, valsize, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    
    splits = generate_train_val(x, y, valsize)
    train_x = splits["train-x"]
    train_y = splits["train-y"]
    val_x = splits["val-x"]
    val_y = splits["val-y"]
    
    logistic.fit(train_x, train_y)

    results = _eval_model(logistic, train_x, train_y, val_x, val_y)

    
    return results


def eval_kfold(x, y, k, logistic):
    
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    fold_assignments = generate_kfold(x, y, k)

    
    train_accs = []
    train_aucs = []
    val_accs = []
    val_aucs = []

    for fold in range(k):
        train_indices = np.where(fold_assignments != fold)[0]
        val_indices = np.where(fold_assignments == fold)[0]

        train_x, train_y = x[train_indices], y[train_indices]
        val_x, val_y = x[val_indices], y[val_indices]

        logistic.fit(train_x, train_y)

        eval_results = _eval_model(logistic, train_x, train_y, val_x, val_y)

        train_accs.append(eval_results["train-acc"])
        train_aucs.append(eval_results["train-auc"])
        val_accs.append(eval_results["val-acc"])
        val_aucs.append(eval_results["val-auc"])

    avg_train_acc = np.mean(train_accs)
    avg_train_auc = np.mean(train_aucs)
    avg_val_acc = np.mean(val_accs)
    avg_val_auc = np.mean(val_aucs)
    results = {"train-acc": avg_train_acc,
               "train-auc": avg_train_auc,
               "val-acc": avg_val_acc,
               "val-auc": avg_val_auc}
    
    
    return results


def eval_mccv(x, y, valsize, s, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    train_accs = []
    train_aucs = []
    val_accs = []
    val_aucs = []

    for _ in range(s):
        splits = generate_train_val(x, y, valsize)
        train_x = splits["train-x"]
        train_y = splits["train-y"]
        val_x = splits["val-x"]
        val_y = splits["val-y"]

        logistic.fit(train_x, train_y)

        eval_results = _eval_model(logistic, train_x, train_y, val_x, val_y)

        train_accs.append(eval_results["train-acc"])
        train_aucs.append(eval_results["train-auc"])
        val_accs.append(eval_results["val-acc"])
        val_aucs.append(eval_results["val-auc"])

    avg_train_acc = np.mean(train_accs)
    avg_train_auc = np.mean(train_aucs)
    avg_val_acc = np.mean(val_accs)
    avg_val_auc = np.mean(val_aucs)
    
    results = {"train-acc": avg_train_acc,
               "train-auc": avg_train_auc,
               "val-acc": avg_val_acc,
               "val-auc": avg_val_auc}
    
    return results



