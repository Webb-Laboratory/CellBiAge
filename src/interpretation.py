import numpy as np
import pandas as pd
from operator import itemgetter


def coefs_list_10(final_data, final_models):
    coefs_list = []
    for i in range(len(final_data)):
        X_test, y_test = final_data[i]
        clf = final_models[i]
    
        y_test_pred = clf.predict(X_test)
        y_test_prob = clf.predict_proba(X_test)
    
        coefs = clf.coef_
        coefs_list.append(coefs)
        
    coefs_list_1 = np.array([a[0] for a in coefs_list])
    mean_coefs = np.mean(coefs_list_1,axis=0)
    std_coefs = np.std(coefs_list_1,axis=0)
    return mean_coefs, std_coefs


def abs_coefs_list(mean_coefs, X):
    abs_coefs = np.absolute(mean_coefs)
    abs_thetas_tuple = [(i,coef) for i,coef in enumerate(abs_coefs)]
    results_abs = sorted(abs_thetas_tuple, key=itemgetter(1), reverse=True)
    print(pd.value_counts(abs_coefs == 0))
    ranked_list = [X.columns[result[0]] for result in results_abs]
    return ranked_list