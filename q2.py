import numpy as np
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
    lr_model = LogisticRegression(max_iter=1000, penalty='none', solver='saga')  # No regularization
    
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

def plot_roc_curves(results, title):
    plt.figure()
    for label, result in results.items():
        fpr, tpr, _ = roc_curve(testy, result['test-prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # Load the training and test data
    train_data = pd.read_csv(train_path, delimiter=' ', header=None, dtype=np.float64).to_numpy()
    test_data = pd.read_csv(test_path, delimiter=' ', header=None, dtype=np.float64).to_numpy()
    trainx, trainy = train_data[:, :-1], train_data[:, -1]
    testx, testy = test_data[:, :-1], test_data[:, -1]
             
    trainx_std, testx_std = do_std(trainx, testx)  
    trainx_nothing, testx_nothing = do_nothing(trainx, testx)
    trainx_log, testx_log = do_log(trainx, testx)
    trainx_bin, testx_bin = do_bin(trainx, testx)
    
    nb_results_std = eval_nb(trainx_std, trainy, testx_std, testy)
    nb_results_nothing = eval_nb(trainx_nothing, trainy, testx_nothing, testy)
    nb_results_log = eval_nb(trainx_log, trainy, testx_log, testy)
    nb_results_bin = eval_nb(trainx_bin, trainy, testx_bin, testy)
    
    
    
    results = {
    'Preprocessing': ['Standardization', 'No Preprocessing', 'Log Transformation', 'Binarization'],
    'Train Accuracy': [nb_results_std['train-acc'], nb_results_nothing['train-acc'], nb_results_log['train-acc'], nb_results_bin['train-acc']],
    'Train AUC': [nb_results_std['train-auc'], nb_results_nothing['train-auc'], nb_results_log['train-auc'], nb_results_bin['train-auc']],
    'Test Accuracy': [nb_results_std['test-acc'], nb_results_nothing['test-acc'], nb_results_log['test-acc'], nb_results_bin['test-acc']],
    'Test AUC': [nb_results_std['test-auc'], nb_results_nothing['test-auc'], nb_results_log['test-auc'], nb_results_bin['test-auc']]
        }

    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results in a table format
    print(results_df.to_string(index=False))          
    
    lr_results_std = eval_lr(trainx_std, trainy, testx_std, testy)
    lr_results_nothing = eval_lr(trainx_nothing, trainy, testx_nothing, testy)
    lr_results_log = eval_lr(trainx_log, trainy, testx_log, testy)
    lr_results_bin = eval_lr(trainx_bin, trainy, testx_bin, testy)
    
    # Create a DataFrame to store the results
    results = {
        'Preprocessing': ['Standardization', 'No Preprocessing', 'Log Transformation', 'Binarization'],
        'Train Accuracy': [lr_results_std['train-acc'], lr_results_nothing['train-acc'], lr_results_log['train-acc'], lr_results_bin['train-acc']],
        'Train AUC': [lr_results_std['train-auc'], lr_results_nothing['train-auc'], lr_results_log['train-auc'], lr_results_bin['train-auc']],
        'Test Accuracy': [lr_results_std['test-acc'], lr_results_nothing['test-acc'], lr_results_log['test-acc'], lr_results_bin['test-acc']],
        'Test AUC': [lr_results_std['test-auc'], lr_results_nothing['test-auc'], lr_results_log['test-auc'], lr_results_bin['test-auc']]
    }

    # Convert the dictionary to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results in a table format
    print(results_df.to_string(index=False))  
    
    nb_results = {
    'Standardization': nb_results_std,
    'No Preprocessing': nb_results_nothing,
    'Log Transformation': nb_results_log,
    'Binarization': nb_results_bin
    }
    plot_roc_curves(nb_results, 'ROC Curves for Naive Bayes Models')
    # Plot ROC curves for Logistic Regression models
    lr_results = {
        'Standardization': lr_results_std,
        'No Preprocessing': lr_results_nothing,
        'Log Transformation': lr_results_log,
        'Binarization': lr_results_bin
    }
    plot_roc_curves(lr_results, 'ROC Curves for Logistic Regression Models')

    # Plot ROC curves for the best Naive Bayes and Logistic Regression models
    best_nb_result = max(nb_results.values(), key=lambda x: x['test-auc'])
    best_lr_result = max(lr_results.values(), key=lambda x: x['test-auc'])
    best_results = {
        'Best Naive Bayes': best_nb_result,
        'Best Logistic Regression': best_lr_result
    }
    plot_roc_curves(best_results, 'ROC Curves for Best Models')            
                                