import segmentation as seg
import pandas as pd
import ydata_profiling
from yellowbrick.features import (Rank2D, RadViz)
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from math import ceil
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def get_database(method:str):
    """
    Forms the database and exports it into a .csv file. Uses the default settings.

    arguments: \n
    None

    returns:\n
    True
    """

    EDA = [4]
    seg.download_dataset()
    database = seg.form_database(directory="data", channel=EDA, data_count=20, segment_length=30, method=method)
    seg.export_database(method + "_segments.csv", database)
    
    return True

def create_profile(df:pd.DataFrame):
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

def pre_prepare_data(df:pd.DataFrame, nan_handling:str="zeroes"):
    """
    Basic preprocessing of the data. Drop all NaN rows and separates the features.
    
    :param df: panda dataframe containing all data
    :param nan_handling: Method to handle NaN values. Can be either "none", "zeroes", "iterative_imputing", "median".

    :return new_X: New panda dataframe without NaN rows with features
    :return y: panda dataframe with the estimator feature
    """
    #basic preprocessing
    y = df.stress
    X = df.drop(columns = ["stress"])
    match nan_handling:
        case "none":
            new_df = df.dropna()
            y = new_df.stress
            new_X = new_df.drop(columns = ["stress"])
        case "zeroes": 
            new_X = X.fillna(0)
        case "iterative_imputing":
            imputer = IterativeImputer(random_state=42)
            new_X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        case "median":
            new_X = X.fillna(df.median())
        case _:
            raise ValueError("nan_handling has to be either 'drop', 'zeroes', 'iterative_imputing' or 'median'")
    
    return new_X, y

def prepare_data(df:pd.DataFrame, random_state:int, nan_handling:str="zeroes"):
    """
    Prepares the data for classification. Drops all NaN rows, separates the 
    infodata and the estimator feature.

    :param df: panda dataframe \n
    :param random_state: number to reproduce or randomize splitting

    :return X_train: panda dataframe of all other relevant features for training \n
    :return y_train: panda series with the estimator feature for training\n
    :return X_test:  panda dataframe of all other relevant features for testing \n
    :return y_test:  panda series with the estimator feature for testing \n
    :return groups:  array with the group indices for cross-validation
    """

    #basic preprocessing
    X,y = pre_prepare_data(df, nan_handling)

    #split the dataset into training and testing sets
    groups = X.subject
    number_of_subjects = X.subject.unique()
    train_size = (len(number_of_subjects) - ceil(0.1 * len(number_of_subjects))) / len(number_of_subjects)
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

def train_model(X_train:pd.DataFrame, y_train:pd.DataFrame, groups:list, model_select:str, random_state:int ):
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
    print("Training", model_select)
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

def evaluate_model(model:str, X_test:pd.DataFrame, y_test:pd.DataFrame):
    """
    Docstring for evaluate_model
    
    :param model: sklearn machine learning model
    :param X_test: panda dataframe of all other relevant features for testing
    :param y_test: panda series with the estimator feature for testing
    
    :return accuracy: accuracy score of the model on the test set
    :return recall: recall score of the model on the test set
    :return precision: precision score of the model on the test set
    :return f1: f1 score of the model on the test set
    """
    y_predict = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)

    return accuracy, recall, precision, f1

def form_evaluation_entry(X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame, groups:list, model_select:str, random_state:int):
    """
    Forms a dictionary with the model, random_state and all relevant evaluation metrics.
    
    :param X_train: panda dataframe of all other relevant features for training
    :param y_train: panda series with the estimator feature for training
    :param X_test: panda dataframe of all other relevant features for testing
    :param y_test: panda series with the estimator feature for testing
    :param groups: array with the group indices for cross-validation
    :param model_select: The machine learning algorithm upon which the data will be trained. Has to be 
    either "knn", "svm", "nb", "lr" or "rf"
    :param random_state: int to add reproducability to the results

    :return evaluation: dictionary with all relevant evaluation metrics
    """
    model = train_model(X_train, y_train, groups, model_select, random_state)
    v = evaluate_model(model, X_test, y_test)
    evaluation = {"model": model_select, "random_state": random_state, "accuracy": v[0], "recall": v[1], "precision": v[2], "f1_score": v[3]}
    return evaluation

def gather_evaluation_metrics(df:pd.DataFrame, random_state:int, repetitions:int, nan_handling:str="zeroes"):
    """
    Runs multiple evaluations of all models with different data splits each time and writes the results  
    into an array of dictionaries.

    :param df: panda dataframe 
    :param random_state: int to add reproduceability to the results
    :param repetitions: number of times the evaluations will be repeated on different data
    """
    evaluation_database =[]
    for i in range(repetitions):
        kwargs = prepare_data(df, random_state+i, nan_handling)
        evaluation_database.append(form_evaluation_entry(*kwargs,"knn", random_state+i))
        evaluation_database.append(form_evaluation_entry(*kwargs,"svm", random_state+i))
        evaluation_database.append(form_evaluation_entry(*kwargs,"nb", random_state+i))
        evaluation_database.append(form_evaluation_entry(*kwargs,"lr", random_state+i))
        evaluation_database.append(form_evaluation_entry(*kwargs,"rf", random_state+i))
    return evaluation_database

def save_evaluation_database(evaluation_database:list, name:str):
    """
    Saves the evaluation database as a .csv file.

    :param evaluation_database: array of dictionaries with evaluation metrics
    :param name: path/name to save the .csv file
    """
    df = pd.DataFrame(evaluation_database)
    df.to_csv(name, index=False)
    return True

def gather_results(method:str, nan_handling:str="zeroes"):
    """
    Gathers results by using machine learning to predict stress with the help of a 
    specified component seperation method.
    
    :param method: The method of component seperation to be used. Has to be "cvxEDA", 
        "smoothmedian', "highpass" or "sparseeda".
    """
    #create segments, use component seperation, establish database
    get_database(method)
    df = pd.read_csv(method + "_segments.csv")

    #evaluate machine learning models
    save_evaluation_database(gather_evaluation_metrics(df, 42, 10, nan_handling),"results/"+ nan_handling + "_filling/" + "model_evaluation/" + method + "_results.csv")

    #get averages and save them
    df_results=pd.read_csv("results/"+ nan_handling + "_filling/" + "model_evaluation/" + method + "_results.csv")
    df_mean = df_results.groupby("model", as_index=False)[["accuracy","recall","precision", "f1_score"]].mean()
    df_mean.to_csv("results/"+ nan_handling + "_filling/" + "model_evaluation/mean_" + method + "_results.csv", index=False)

    #plot a boxplot to show the results
    fig, ax = plt.subplots(figsize=(12,6))
    df_results.boxplot(by='model', column=["accuracy"],ax=ax)
    fig.savefig("results/" + nan_handling + "_filling/"+ method + "_boxplot.svg", dpi=300, format="svg", bbox_inches="tight")

    return True

def evaluate_component_separation():
    """
    Evaluates component seperation by using models provided by neurokit.
    """
    #zero filling for NaN values
    #gather_results("cvxEDA", nan_handling="zeroes")
    #gather_results("smoothmedian", nan_handling="zeroes")
    #gather_results("highpass", nan_handling="zeroes")

    #iterative imputing for NaN values
    #gather_results("cvxEDA", nan_handling="iterative_imputing")
    #gather_results("smoothmedian", nan_handling="iterative_imputing")
    #gather_results("highpass", nan_handling="iterative_imputing")

    #averages for NaN values
    #gather_results("cvxEDA", nan_handling="median")
    #gather_results("smoothmedian", nan_handling="median")
    #gather_results("highpass", nan_handling="median")

    #dropping all NaN values
    gather_results("cvxEDA", nan_handling="none")
    gather_results("smoothmedian", nan_handling="none")
    gather_results("highpass", nan_handling="none")
    return True

def plot_correlation(X,y,name):
    """
    Plots a correlation heatmap and saves it under the given path/name.

    arguments: \n
    :param X: panda dataframe, which will be examined \n
    :param y: panda dataframe, not important
    :param name: path/name to save the plot

    returns: \n
    None
    """

    fig,ax = plt.subplots(figsize = (12,12))
    pcv = Rank2D(features=X.columns, algorithm ="pearson")
    pcv.fit(X,y)
    pcv.transform(X)
    pcv.poof()
    fig.savefig(name,dpi=300, format = "svg", bbox_inches="tight")
    return True

def plot_radviz(X, y, name):
    """
    Reads a panda dataframe and plots a RadViz plot and saves it under the given path/name.

    :param X: panda dataframe with the features
    :param y: panda dataframe with the estimator feature
    :param name: path/name to save the plot
    """
    #fix indices for yellowbrick
    X = X.drop(columns="subject")
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    #plot the radviz
    fig,ax = plt.subplots(figsize = (12,12))
    rv = RadViz(classes=["no_stress","stress"], features=X.columns, ax=ax)
    rv.fit(X,y)
    rv.show()
    fig.savefig(name, dpi=300, format="svg",bbox_inches ="tight")
    return True

def plot_confusion_matrix(model, X_test, y_test, name):
    """
    Plots a confusion matrix for a given model and saves it under the given path/name.
    
    :param model: the model to be evaluated on the new dataset
    :param X_test: new feature dataset
    :param y_test: new estimator feature dataset
    :param name:   path/name to save the plot
    """
    mapping = {0: "no stress", 1: "stress"}
    fig,ax = plt.subplots(figsize=(12,12))
    cm_viz = ConfusionMatrix(model, classes=["no stress", "stress"], encoder=mapping, force_model=True)
    cm_viz.score(X_test, y_test)
    cm_viz.show()
    fig.savefig(name, dpi=300, format="svg", bbox_inches="tight")
    return True


#for data vizualisation
#new_df, X, y = pre_prepare_data(df)
#plot_radviz(X, y, "results/radviz.svg")
#plot_correlation(X.drop(columns=["subject"]),y,"results/correlation.svg")
#create_profile(X)
#plot_confusion_matrix(best_model, X_test, y_test, "results/confusion_matrix.svg")

#paper results
#gather_results("cvxEDA")

#component seperation results
evaluate_component_separation()




    

