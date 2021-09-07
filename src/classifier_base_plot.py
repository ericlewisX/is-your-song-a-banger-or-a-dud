# Imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.inspection import permutation_importance

# Plot Functions

def PlotRocCurve(model, X_test, y_test, **kwargs):
    ''' 
    The function will return the ROC curver plot.
        
    Parameters
    -----------
    model : The model for which you want an ROC curve.
    
    X_test : X_test set.
    
    y_test : y_test set.
    
    **kwargs : Any additional parameters given by `sklearn.metrics`'s `plot_roc_curve` function.
    
    Returns
    --------
    plot
    
    '''
    return plot_roc_curve(model, X_test, y_test, **kwargs)


def PlotConfusionMatrix(model, X_test, y_test):
    ''' 
    The function will return the confusion matrix plot.
        
    Parameters
    -----------
    model : The model for which you want an ROC curve.
    
    X_test : X_test set.
    
    y_test : y_test set.
     
    Returns
    --------
    plot
    
    '''
    return plot_confusion_matrix(model, X_test, y_test)


def PermutationImportance(model, feature_names, X_test, y_test, **kwargs):
    ''' 
    The function will return the ROC curver plot.
        
    Parameters
    -----------
    model : The model for which you want an ROC curve.
    
    feature_names : The column names of the features you are inquiring about.
    
    X_test : X_test set.
    
    y_test : y_test set.
    
    **kwargs : Any additional parameters given by `sklearn.inspection`'s `permutation_importance` function.
    
    Returns
    --------
    plot
    
    '''
    r = permutation_importance(model, X_test, y_test, n_repeats=30, **kwargs)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]:<8}"
                f"{r.importances_mean[i]:.3f}"
                f" +/- {r.importances_std[i]:.3f}")

    fig, ax = plt.subplots(1, figsize=(8, 8))
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh', ax=ax)
    fig