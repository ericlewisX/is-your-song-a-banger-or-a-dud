# Imports #
import pandas as pd
import numpy as np

# Machine Learning
import classifier_base as CLFScores

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.tree import RandomForestClassifier
from sklearn.inspection import permutation_importance


# Functions

def ModelRF(X, y, **kwargs):
    '''
    This function takes in a feature matrix X (scales it down using MinMaxScaler()), and a target array y, and builds a Random Forest model using them.

    Parameters
    ----------
    X : pandas Array or pandas Dataframe
        [Feature matrix]
    y : pandas Series
        [Target array]
    **kwargs : kwargs for `RandomForestClassifier()`

    Returns
    -------
    clf : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000.]
    X_test : numpy array
        [X_test made from train_test_split]
    y_test : pandas Series
        [y_test made from train_test_split]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size = 0.8)
    clf = RandomForestClassifier(**kwargs, max_iter = 1000)
    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def ScoresRF(model, X_test, y_test):
    '''
    This function takes in our fitted model and returns the performace metrics Accuracy, Precision, Recall. 

    Parameters
    ----------
    model : sklearn.tree._classes.RandomForestClassifier
        [Fitted Random Forest Model built with a 0.80 training size, and max_iter 1000]
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
    
    print(f"- - - - Random Forest Model - - - -")
    print(f"Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}")

    return accuracy, precision, recall


def OptimizeRF(X, y, **kwargs):
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
    rf_random.best_estimator_ : sklearn.tree._classes.RandomForestClassifier
        [The best Random Forest Model after hyper-parameter tuning.]
    rf_random.best_params_ : dict
        [The parameters for the best model.]
    result : sklearn.utils.Bunch
        [A Bunch for the permutation importance of the tuned model.]
    '''
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(scale, y, train_size=0.8, **kwargs)
    
    ###Random Hyperparameter Grid
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 15, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    
  

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features, 
                   'max_depth': max_depth, 
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier(bootstrap = True, oob_score=True)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = random_grid,
                                   n_iter = 5, cv = 5, verbose=2, 
                                   random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=0)
    
    return rf_random.best_estimator_, rf_random.best_params_, result
