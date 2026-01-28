import segmentation as seg
import model_evaluation as me
import pandas as pd

#for data vizualisation
##whole signal decomposition
#signal, sampling_frequency, annotation_points = seg.extract_signal("data/Subject1_AccTempEDA", EDA)
#components, cleaned_signal, _, __, ___ = seg.components_separation(signal, method="cvxEDA", fs=sampling_frequency)
#seg.plot_segment("results/segment_decomposition.svg", cleaned_signal, components[1], components[0], "EDA Signal Decomposition - Subject 1")

##plot radviz, correlation, confusion matrix and create profile
#database = seg.form_database(directory="data", channel=[4], data_count=20, segment_length=30, method="cvxEDA")
#seg.export_database("cvxEDA_segments.csv", database)
#df=pd.read_csv("cvxEDA_segments.csv")
#X, y = me.pre_prepare_data(df)
#me.plot_radviz(X, y, "results/radviz.svg")
#me.plot_correlation(X.drop(columns=["subject"]),y,"results/correlation.svg")
#me.create_profile(X)

#paper results
#me.gather_results("cvxEDA", nan_handling="zeroes")

#component seperation results
#me.evaluate_component_separation()