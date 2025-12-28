import segmentation as seg
import pandas as pd
import ydata_profiling
from yellowbrick.features import Rank2D
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy as np
from math import ceil
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_database():
    """
    Forms the database and exports it into a .csv file. Uses the default settings.

    arguments: \n
    None

    returns:\n
    True
    """

    EDA = [4]
    database = seg.form_database(directory="data", channel=EDA, data_count=20, segment_length=30)
    seg.export_database("segments.csv", database)
    
    return True

def create_profile(df):
    """
    Creates a panda profile with correlation and missing data statistics.
    The profile is then exported as a HTML file to be conveniently read
    in the browser. 

    arguments: \n
    df - panda dataframe

    returns:\n
    True
    """

    profile = ydata_profiling.ProfileReport(df,title = "ProfilingReport")
    profile.to_file("profile.html")

    return True

def prepare_data(df, random_state):
    """
    Prepares the data for classification. Drops all NaN rows, separates the 
    infodata and the classification feature.

    :param df: panda dataframe \n
    :param random_state: number to reproduce or randomize splitting

    :return X_train: panda dataframe of all other relevant features for training \n
    :return y_train: panda dataframe with the classification feature for training\n
    :return X_test:  panda dataframe of all other relevant features for testing \n
    :return y_test:  panda dataframe with the classification feature for testing \n
    :return groups:  array with the group indices for cross-validation
    """

    #basic preprocessing
    df1 = df.dropna()
    y = df1.stress
    X = df1.drop(columns = ["stress"])

    #split the dataset into training and testing sets
    groups = df1.subject
    train_size = (len(df.subject.unique()) - ceil(0.1 * len(df.subject.unique()))) / len(df.subject.unique())
    gss = ms.GroupShuffleSplit(n_splits=1, train_size=train_size , random_state=random_state)
    train_idx, test_idx = next(gss.split(X,y, groups=groups))

    #allocate the datasets
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    #extract the groups for cross-validation and drop the subject information
    groups = X_train.subject
    X_test = X_test.drop(columns="subject")
    X_train = X_train.drop(columns="subject")
    y_test = y_test.drop(columns="subject")
    y_train = y_train.drop(columns="subject")

    return X_train, y_train, X_test, y_test, groups

def train_model(X_train, y_train, groups, model_select, random_state ):
    """
    Train a model on a training set and validate it. The specifing model can be 
    set using the model_select variable. Cross-validation will be based upon a 
    LeaveOneGroupOut(here in fact GSS) method due to the data not being i.i.d.
    Returns a trained model.
    
    :param X_train: Panda dataframe training set with features
    :param y_train: Panda dataframe training set with estimator feature
    :param groups: Array for the cross-validation
    :param model_select: The machine learning algorithm upon which the data will be trained. Has to be 
    either "knn", "svm", "nb", "lr" or "rf"
    :param random_state: int to add reproducability to the results

    :return best_model: The trained model
    """
    unique_subjects = len(groups.unique())
    train_size = (unique_subjects - 1) / unique_subjects 
    gss = ms.GroupShuffleSplit(n_splits=unique_subjects, train_size=train_size, random_state=random_state)

    match model_select:
        case "knn": #K Nearest Neighbors
            #initialize the model and the hyperparameter grid
            predefined_model = KNeighborsClassifier(algorithm="auto", leaf_size=30, metric="minkowski", 
                                       n_neighbors=5, p=2, weights="uniform")
            param_grid={
                "n_neighbors":  [3,5,7,9,11,13,15],
                "weights": ["uniform", "distance"],
            }
            model = KNeighborsClassifier()

        case "svm": #Support Vector Machines
            #initilize the model and hyperparameter grid
            predefined_model = SVC(C=1.0, cache_size=1024, class_weight=None, coef0=0.0, 
                      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf', 
                      max_iter=-1, probability=True, random_state=random_state, 
                      shrinking=True, tol=0.001, verbose=False )
            param_grid={
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "kernel": ["linear", "rbf", "poly"],
                "class_weight": [None, "balanced"],
            }
            model = SVC(random_state=random_state)
        
        case "nb": #Naive Bayes
            #initilize the model and hyperparameter grid
            predefined_model = GaussianNB(priors=None, var_smoothing=1e-09)
            param_grid={
                "var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
            }
            model = GaussianNB()

        case "lr": #Logistic Regression
            #initilize the model and hyperparameter grid
            predefined_model = LogisticRegression(C=1.0, class_weight=None,dual=False, fit_intercept=True, 
                                    intercept_scaling=1, max_iter=100,random_state=random_state, solver='liblinear',tol=0.0001, 
                                    verbose=0, warm_start=False)
            param_grid={
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "class_weight": [None, "balanced"],

            }
            model = LogisticRegression(random_state=random_state)

        case "rf": #RandomForest
            #initialize the model and the hyperparameter grid
            predefined_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None, 
                                        max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                        min_samples_leaf=1, min_samples_split=2, 
                                        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, 
                                        random_state=42, verbose=0, warm_start=False)
            param_grid={
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [None, 10, 20, 40],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.3, 0.5],
            }
            model = RandomForestClassifier(random_state=random_state)
        
        case _: #wrong model selected
            return False 
        
    #calculate prefitting scores, perform a randomized search; change the n_iter parameter if needed
    prefitting_scores = ms.cross_val_score(predefined_model, X_train, y_train, groups=groups, cv=gss)
    print("Prefitting Scores:", prefitting_scores)
    gs = ms.RandomizedSearchCV(estimator=model, 
                                param_distributions=param_grid, n_iter=30, cv=gss, 
                                scoring='accuracy', n_jobs=-1, random_state=random_state)
    gs.fit(X_train, y_train, groups=groups)

    #train the 'best' estimator and calculate postifitting scores
    best_model = gs.best_estimator_
    best_model.fit(X_train, y_train)
    postfitting_scores = ms.cross_val_score(best_model, X_train, y_train, groups=groups, cv=gss)
    print("Postfitting Scores:", postfitting_scores)
    return best_model

def plot_correlation(X,y,name):
    """
    Plots a correlation heatmap and saves it under the given path/name.

    arguments: \n
    :param X: panda dataframe, which will be examined \n
    :param y: panda dataframe, not important

    returns: \n
    None
    """

    fig,ax = plt.subplots(figsize = (6,6))
    pcv = Rank2D(features=X.columns, algorithm ="pearson")
    pcv.fit(X,y)
    pcv.transform(X)
    pcv.poof()
    fig.savefig(name,dpi=300, format = "svg", bbox_inches="tight")

df = pd.read_csv("segments.csv")
X_train, y_train, X_test, y_test, groups = prepare_data(df,42)
#create_profile(X_train)
train_model(X_train, y_train, groups, "nb",42)