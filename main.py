import segmentation as seg
import model_evaluation as me

#for data vizualisation
#X, y = me.pre_prepare_data(df)
#me.plot_radviz(X, y, "results/radviz.svg")
#me.plot_correlation(X.drop(columns=["subject"]),y,"results/correlation.svg")
#me.create_profile(X)
#me.plot_confusion_matrix(best_model, X_test, y_test, "results/confusion_matrix.svg")

#paper results
#me.gather_results("cvxEDA")

#component seperation results
me.evaluate_component_separation()