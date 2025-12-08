import segmentation as seg
import pandas as pd
import ydata_profiling
from yellowbrick.features import Rank2D
import matplotlib.pyplot as plt
import sklearn.model_selection
import numpy as np
from math import ceil

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

    arguments: \n
    df - panda dataframe \n
    random_state - number to reproduce or randomize splitting

    returns:\n
    X_train - panda dataframe of all other relevant features for training \n
    y_train - panda dataframe with the classification feature for training\n
    X_valid - panda dataframe for validation \n
    y_valid - panda dataframe for validation \n
    X_test - panda dataframe of all other relevant features for testing \n
    y_test - panda dataframe with the classification feature for testing
    """

    #basic preprocessing
    df1 = df.dropna()
    y = df1.stress
    X = df1.drop(columns = ["stress"])

    #first splitting into training and test
    groups = df1.subject
    train_size = (len(df.subject.unique()) - ceil(0.1 * len(df.subject.unique()))) / len(df.subject.unique())
    print(train_size)
    gss = sklearn.model_selection.GroupShuffleSplit(n_splits=1, train_size=train_size , random_state=random_state)
    train_idx, test_idx = next(gss.split(X,y, groups=groups))

    X_other = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_other = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    X_test.drop(columns="subject")
    y_test.drop(columns="subject")

    #split further into validation and training data
    training_subjects = X_other.subject.unique()
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(training_subjects)

    val_subjects = training_subjects[:1]
    other_subjects = training_subjects[1:]

    val_mask = X_other.subject.isin(val_subjects)
    train_mask = X_other.subject.isin(other_subjects)

    X_valid = X_other[val_mask].drop(columns="subject")
    X_train = X_other[train_mask].drop(columns="subject")
    y_valid = y_other[val_mask].drop(columns="subject")
    y_train = y_other[train_mask].drop(columns="subject")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def plot_correlation(X,y,name):
    """
    Plots a correlation heatmap and saves it under the given path/name.

    arguments: \n
    X - panda dataframe, which will be examined \n
    y - panda dataframe, not important

    returns: \n
    None
    """

    fig,ax = plt.subplots(figsize = (6,6))
    pcv = Rank2D(features=X.columns, algorithm ="pearson")
    pcv.fit(X,y)
    pcv.transform(X)
    pcv.poof()
    fig.savefig(name,dpi=300, bbox_inches="tight")

df = pd.read_csv("segments.csv")
X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data(df,42)
create_profile(X_test)
