# Imports
import numpy as np
import pandas as pd

# Machine Learning
import classifier_base as CLFScores

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance


# Functions

def ModelLR(X, y, **kwargs):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a logistic regression model using them.

    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `LogisticRegression()`

    Returns
    -------
    clf : sklearn.linear_model._logistic_LogisticRegression
        [Fitted Logistic Regression Model built with a 0.80 training size, and max_iter 1000.]
    X_test : numpy array
        [X_test made from train_test_split]
    y_test : pandas Series
        [y_test made from train_test_split]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
   
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size = 0.8)
    clf = LogisticRegression(**kwargs, max_iter = 1000)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def ScoresLR(model, X_test, y_test):
    '''
    This function takes in our fitted model and returns the performace metrics Accuracy, Precision, Recall. 

    Parameters
    ----------
    model : sklearn.linear_model._logistic_LogisticRegression
        [Fitted Logistic Regression Model built with a 0.80 training size, and max_iter 1000]
    X_test : numpy array
        [X_test made from train_test_split]
    y_test : pandas Series
        [y_test made from train_test_split]

    Returns
    -------
    accuracy : numpy Float64
        [Performance metric score for accuracy]
    precision : numpy Float64
        [Performance metric score for precision]
    recall : numpy Float64
        [Performance metric score for recall]
    '''
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = recall_score(y_test, predictions)
    recall = precision_score(y_test, predictions)
    
    print(f"- - - - Logistic Regression Model - - - -")
    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")

    return accuracy, precision, recall


def OptimizeLR(X, y, **kwargs):
    '''
    OptimizeLR [summary]

    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `train_test_split`

    Returns
    -------
    lr_random.best_estimator_ : sklearn.linear_model._logistic.LogisticRegression
        [The best Logistic Regression Model after hyper-parameter tuning.]
    lr_random.best_params_ : dict
        [The parameters for the best model.]
    result : sklearn.utils.Bunch
        [A Bunch for the permutation importance of the tuned model.]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size = 0.8, **kwargs)
    
    ### Random Hyperparameter Grid
    penalty = ['l2']
    C = [2, 5, 10, 15, 20]
    
    # Create the random grid
    random_grid = {'penalty': penalty, 'C': C}
      
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    lr = LogisticRegression(max_iter = 1000)
    
    # Random search of parameters, using 5 fold cross validation.
    lr_random = RandomizedSearchCV(estimator = lr,
                                   param_distributions = random_grid,
                                   n_iter = 5, 
                                   cv = 5, 
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs = 1)
    
    # Fit the random search model
    clf = lr_random.fit(X_train, y_train)
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
   
    return lr_random.best_estimator_, lr_random.best_params_, result
