# Data Wrangling/Cleaning
import pandas as pd
import numpy as np

from datetime import datetime
from six import StringIO

# Custom Functions
import classifier_base as CLFScores
from DecisionTreeCLF import ModelDT, ScoresDT, OptimizeDT
from LogisticRegressionCLF import ModelLR, ScoresLR, OptimizeLR
from RandomForestCLF import ModelRF, ScoresRF, OptimizeRF

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, plot_confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier

# Visualizations
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go

# Saving Model
import pickle5


if __name__ == "__main__":
    
    # Loading Clean Data
    cleaned_df = pd.read_csv("cleaned_and_balanced_top100", index_col = 0)
    
    # Extra Data Manipulation
    cleaned_df['release_date'] = pd.to_datetime(cleaned_df['release_date'])
    cleaned_df['WeekID'] = pd.to_datetime(cleaned_df['WeekID'])
    
    print("Time Interval for Dataset Songs on the Hot 100 Chart:", cleaned_df['WeekID'].min(), cleaned_df['WeekID'].max())
    
    # Sub-dataframes
    hot100 = df[:24185]
    hot100.describe()
    
    sns.pairplot(hot100, vars=['danceability', 'energy', 'key','loudness','mode','speechiness',
                           'acousticness','instrumentalness','liveness','valence','tempo','time_signature'], corner=True)
    sns.pairplot(hot100, vars=['Peak Position', 'Weeks on Chart','danceability', 'energy', 'key','loudness'], corner=True)
    sns.pairplot(hot100, vars=['Peak Position', 'Weeks on Chart','mode','speechiness',
                           'acousticness','instrumentalness','liveness','valence','tempo','time_signature'], corner=True)
    sns.pairplot(hot100, vars=['Peak Position', 'Weeks on Chart', 'key','loudness','mode','speechiness',
                           'acousticness','instrumentalness','liveness','valence','tempo','time_signature'])
    
    # Feature and Target Matrices
    X = cleaned_df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo']]
    y = cleaned_df['top100'].copy()
    y_df = pd.DataFrame(y)
    
    # X.sample(5)
    # y.sample(5)
    
    y_df[y_df['top100'] == 1] = 'True'
    y_df[y_df['top100'] == 0] = 'False'
    
    
    # Balanced Data plot
    f, ax = plt.subplots(figsize=(5,6))

    sns.countplot(y_df['top100'], data=y_df, palette="Set1").set(title='Number of Songs', 
                                                                 xlabel='Member of The Hot 100',
                                                                ylabel= 'Number of Unique Songs')
    plt.tight_layout()
    plt.savefig("balanced_df.png")
    plt.show()
    
    
    ################
    ### Modeling ###
    ################
    
    print(X.shape, y.shape)

    ## Logistic Regression
    
    Lmodel, LX_test, Ly_test = ModelLR(X, y, random_state = 42)
    Laccuracy, Lprecision, Lrecall = ScoresLR(Lmodel, LX_test, Ly_test)
    Lbest_model, Lbest_params, result = OptimizeLR(X, y, random_state = 42)
    
    print(Lbest_model, Lbest_model.coef_)
    Ly_pred = Lbest_model.predict(LX_test)
    
    coefX = pd.DataFrame(Lbest_model.coef_, columns = X.columns).transpose()
    coefX.rename(columns = {0 : 'Features'}, inplace = True)
    # coefX
    
    ## Logistic Regression Coefficients plot
    fig = px.bar(coefX, y = 'Features', title = "Logistic Regression Best Model Coefficients")
    fig.update_traces(texttemplate = ['0.385812', '-0.938297', '2.914513', '-1.861425', '-1.470924','-0.382257', 
                                    '0.551609','-0.200651'], textposition = 'outside')
    fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide')
    fig.show()
    
    ## LR Best Model
    print("LR Model Best Parameters : "Lbest_params)
    ScoresLR(Lbest_model, LX_test, Ly_test)
    
    ## LR Plots
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Lbest_model, LX_test, Ly_test, cmap = 'Purples', display_labels = class_names, colorbar = False)
    plt.title("LR Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionLR.png")
    plt.show()
    
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Lbest_model, LX_test, Ly_test, cmap = 'Purples', display_labels = class_names, colorbar = False, normalize = 'true')
    plt.title("LR Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionnormLR.png")
    plt.show()

    print(classification_report(Ly_test, Ly_pred, digits = 3))
    
    ### --- ###
    
    ## Decision Tree
    
    Dmodel, DX_test, Dy_test = ModelDT(X, y, random_state = 42)
    Daccuracy, Dprecision, Drecall = ScoresDT(Dmodel, DX_test, Dy_test)
    Dbest_model, Dbest_params = OptimizeDT(X, y, random_state = 42)
    
    print(Dbest_model)
    Dy_pred = Dbest_model.predict(DX_test)
    
    ## DT Best Model
    print("DT Model Best Parameters : "Dbest_params)
    ScoresDT(Dbest_model, DX_test, Dy_test)
    
    ## DT Plots
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Dbest_model, DX_test, Dy_test, cmap = 'Blues', display_labels = class_names, colorbar = False)
    plt.title("DT Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionDT.png")
    plt.show()
    
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Dbest_model, DX_test, Dy_test, cmap = 'Blues', display_labels = class_names, colorbar = False, normalize = 'true')
    plt.title("DT Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionnormDT.png")
    plt.show()
    
    CLFScores.PlotRocCurve(Dbest_model, DX_test, Dy_test, name = "Decision Tree")
    plt.title("Decision Tree ROC Curve")
    plt.tight_layout()
    plt.savefig("rocDT.png")
    plt.show()
    
    CLFScores.PermutationImportance(Dbest_model, X.columns, DX_test, Dy_test)
    plt.title("Song Permutation Importance")
    plt.tight_layout()
    plt.savefig("DTFI.png")
    plt.show()

    print(classification_report(Dy_test, Dy_pred, digits=3))
    
    
    ### --- ###
    
    ## Random Forest
    
    Rmodel, RX_test, Ry_test = ModelRF(X, y, random_state = 42)
    Raccuracy, Rprecision, Rrecall = ScoresRF(Rmodel, X_test, y_test)
    Rbest_model, Rbest_params, Rmodel = OptimizeRF(X, y, random_state = 42)
    
    print(Rbest_model)
    Ry_pred = Rbest_model.predict(RX_test)
    
    ## RF Best Model
    print("RF Model Best Parameters : "Rbest_params)
    ScoresRF(Rbest_model, RX_test, Ry_test)
    
    ## RF Plots
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Rbest_model, RX_test, Ry_test, cmap = 'Greys', display_labels = class_names, colorbar = False)
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionRF.png")
    plt.show()
    
    class_names = ["Didn't Make It", 'In the Hot 100']
    plot_confusion_matrix(Rbest_model, RX_test, Ry_test, cmap = 'Greys', display_labels = class_names, colorbar = False, normalize = 'true')
    plt.title("RF Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusionnormRF.png")
    plt.show()
    
    CLFScores.PlotRocCurve(Rbest_model, RX_test, Ry_test, name = "Decision Tree")
    plt.title("Random Forest ROC Curve")
    plt.tight_layout()
    plt.savefig("rocRF.png")
    plt.show()
    
    
    CLFScores.PermutationImportance(Rbest_model, X.columns, RX_test, Ry_test)
    plt.title("Song Permutation Importance")
    plt.tight_layout()
    plt.savefig("rfFI.png")
    plt.show()

    print(classification_report(Ry_test, Ry_pred, digits = 3))
    
    #################################
    ### Comparison Between Models ###
    #################################

    print(" ------ ")
    print(" LR Best Model Scores ")
    ScoresLR(Lbest_model, LX_test, Ly_test)
    print(" ------ ")
    print(" DT Best Model Scores ")
    ScoresDT(Dbest_model, DX_test, Dy_test)
    print(" ------ ")
    print(" RF Best Model Scores ")
    ScoresRF(Rbest_model, RX_test, Ry_test)
    print(" ------ ")
    
    