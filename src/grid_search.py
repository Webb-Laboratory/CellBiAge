from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle

def pr_auc_score(y_true, y_scores):
    '''
    Generates the Area Under the Curve for precision and recall.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def ML_pipeline_GridSearchCV(X_other, y_other, X_test, y_test, clf, param_grid, 
                             random_state, custom_cv, pr_auc_scorer, xgbc=False, set_random=True):
    '''
    clf: classifier
    param_grid: parameter grid
    '''
    # create a test set based on groups
    random_state = 42*random_state    
    
    test_X, test_y = shuffle(X_test, y_test, random_state=random_state)
    
    if xgbc:
        clf.set_params(seed=random_state)
    if set_random:
        clf.set_params(random_state=random_state)
        
    # create the pipeline: supervised ML method
    pipe = make_pipeline(clf)
    
    # prepare gridsearch
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring=pr_auc_scorer,
                        cv=custom_cv, return_train_score=True, n_jobs=-1)
    
    grid.fit(X_other, y_other)
    return grid, grid.score(test_X, test_y)