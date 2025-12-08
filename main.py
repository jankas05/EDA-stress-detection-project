import segmentation as seg
import pandas as pd
import ydata_profiling
from yellowbrick.features import Rank2D
import matplotlib.pyplot as plt
import sklearn.model_selection

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
    y_test - panda dataframe with the classification feature for testing\n
    y_train - panda dataframe with the classification feature for training\n
    X_test - panda dataframe of all other relevant features for testing \n
    X_train - panda dataframe of all other relevant features for training 
    """

    df1 = df.dropna()
    y = df1.stress
    X = df1.drop(columns = ["stress"])
    groups = df1.subject

    gss = sklearn.model_selection.GroupShuffleSplit(n_splits=1, train_size=17/20 , random_state=random_state)
    train_idx, test_idx = next(gss.split(X,y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return y_test, y_train, X_test, X_train

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
y_test, y_train, X_test, X_train = prepare_data(df,42)

