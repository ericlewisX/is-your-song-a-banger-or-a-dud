# Imports #
import pandas as pd
import numpy as np

# Machine Learning
import classifier_base as CLFScores

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance


# Functions

def ModelDT(X, y, **kwargs):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Decision Tree model using them.

    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `DecisionTreeClassifier()`

    Returns
    -------
    clf : sklearn.tree._classes.DecisionTreeClassifier
        [Fitted Decision Tree Model built with a 0.80 training size, and max_iter 1000.]
    X_test : numpy array
        [X_test made from train_test_split]
    y_test : pandas Series
        [y_test made from train_test_split]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size = 0.80)
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def ScoresDT(model, X_test, y_test):
    '''
    This function takes in our fitted model and returns the performace metrics Accuracy, Precision, Recall. 

    Parameters
    ----------
    model : sklearn.tree._classes.DecisionTreeClassifier
        [Fitted Decision Tree Model built with a 0.80 training size, and max_iter 1000]
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

    print(f"- - - - Decision Tree Model - - - -")
    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")

    return accuracy, precision, recall


def OptimizeDT(X, y, **kwargs):
    '''
    This function conducts the hyper-parameter tuning step.

    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `train_test_split`

    Returns
    -------
    dtm_random.best_estimator_ : sklearn.tree._classes.DecisionTreeClassifier
        [The best Decision Tree Model after hyper-parameter tuning.]
    dtm_random.best_params_ : dict
        [The parameters for the best model.]
    result : sklearn.utils.Bunch
        [A Bunch for the permutation importance of the tuned model.]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size = 0.80, **kwargs)

    ### Random Hyperparameter Grid
    optimizers = {'max_depth': [3, 5, None], 
                  'min_samples_split': [2, 10], 
                  'max_features': ['sqrt', 'log2', 2, 5, None]}
    
    # Grid search of parameters.
    dtm = GridSearchCV(DecisionTreeClassifier(**kwargs), optimizers, verbose=3)
    
    # Fit the random search model
    clf = dtm.fit(X_train, y_train)
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
    
    return dtm.best_estimator_, dtm.best_params_, result
