import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from tqdm import tqdm
import pickle
import os
import tensorflow as tf
import numpy as np


class Baseline_XGB:
    model_path = '../results/xgbc_best.save'

    def __init__(self, data_path, k_fold=5, need_train=False):
        self.model = XGBClassifier(use_label_encoder=False)
        self.param_grid = {'xgbclassifier__max_depth': [10, 15],
                           "xgbclassifier__learning_rate": [0.08, 0.15],
                           'xgbclassifier__colsample_bytree': [0.7, 0.8],
                           'xgbclassifier__subsample': [0.66],
                           'xgbclassifier__eval_metric': ['logloss']}
        self.data_path = data_path
        self.need_train = need_train
        self.k_fold = k_fold

    def train(self):
        if self.need_train:
            df = pd.read_csv(self.data_path)
            X, y = np.array(df.iloc[:, 1:-4]).astype(dtype=np.float32), np.array(df["target"]).astype(dtype=np.float32)


            models, scores = self.start_GridSearch(X, y)
            best_model = self.find_best(models, scores)
            self.save_model(best_model)

            self.model = best_model
        else:
            with open(self.model_path, 'rb') as handle:
                self.model = pickle.load(handle)

    def start_GridSearch(self, X, y):
        models = []
        scores = []
        for i in tqdm(range(2)):
            grid, test_score = self.ML_pipeline_GridSearchCV(X, y, self.model, self.param_grid, i, self.k_fold)
            print(grid.best_params_)
            print('best CV score:', grid.best_score_)
            print('test score:', test_score)
            models.append(grid.best_estimator_)
            scores.append(test_score)

        return models, scores

    def find_best(self, models, scores):
        '''
        data = {"model": [models[i][-1] for i in range(2)], "scores": scores}
        result = pd.DataFrame(data=data)
        result["model"] = result["model"].astype(str)
        average_performance = result.groupby(["model"]).mean()
        average_performance = average_performance.reset_index()
        return eval(average_performance.loc[average_performance['scores'].idxmax()][0])
        '''
        return models[scores.index(max(scores))]

    def save_model(self, best_model):
        parent_dir, file = os.path.split(self.model_path)
        os.makedirs(parent_dir, exist_ok=True)

        with open(self.model_path, 'wb') as handle:
            pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def ML_pipeline_GridSearchCV(self, X, y, clf, param_grid, random_state, n_folds):
        '''
        clf: classifier
        param_grid: parameter grid
        '''
        # create a test set based on groups
        random_state = 42 * random_state
        X_other, X_test, y_other, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        # onehotencoder for the categorical feature -- tree.ident

        cont_ftrs = list(X_other)[1:]

        # standard scaler for continuous feature
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, cont_ftrs)
            ])

        clf.set_params(seed=random_state)

        # create the pipeline: preprocessor + supervised ML method
        pipe = make_pipeline(preprocessor, clf)

        # prepare gridsearch
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring=make_scorer(accuracy_score),
                            cv=kf, return_train_score=True, n_jobs=-1)

        # do kfold CV on _other
        grid.fit(X_other, y_other)

        return grid, grid.score(X_test, y_test)

    def test(self, X, y):
        y_test_pred = self.model.predict(X)
        m = tf.keras.metrics.BinaryAccuracy()
        m.update_state(y, y_test_pred)
        print(m.result().numpy())



