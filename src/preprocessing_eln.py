import pandas as pd
import tqdm
from tqdm import tqdm
from src.grid_search import *
from sklearn.linear_model import LogisticRegression
from statistics import mean, stdev
import pickle


def assign_target(input_df):
    input_df = pd.read_csv(input_df, index_col=0)
    input_df['animal'] = input_df.index.str[-1]
    input_df['target'] = ((input_df['animal'] =='3')|(input_df['animal']=='4')).astype(int)
    return input_df

def train_test_split(input_train, input_test, binarization=False):
    df_test = assign_target(input_test)
    test_X = df_test.iloc[:,:-2]
    test_y = df_test.target
    test_X, test_y = shuffle(test_X, test_y, random_state=42)

    df_train = assign_target(input_train)
    train = df_train.reset_index()
    train_X = train.iloc[:,1:-2]
    train_y = train.target
    
    if binarization==True:
        test_X = binarize_data(test_X)
        train_X = binarize_data(train_X)
    
    return train_X, train_y, test_X, test_y

def runs_10(train_X, train_y, test_X, test_y, eln_c, l1_ratio, preprocess_method):
    scores = []
    final_test = []
    final_models = []
    for i in tqdm(range(10)):
        random_state = 42*i    
        X_test, y_test = shuffle(test_X, test_y, random_state=random_state)
        X_train, y_train = shuffle(train_X, train_y, random_state=random_state)
    
        eln = LogisticRegression(penalty='elasticnet', C=eln_c, l1_ratio=l1_ratio, 
                                 solver='saga', max_iter=10000000)
        
        eln.fit(X_train, y_train)
    
        y_pred = eln.predict_proba(X_test)[:, 1]
        auprc = pr_auc_score(y_test, y_pred)
    
        final_test.append((X_test, y_test))
        final_models.append(eln)
        scores.append(auprc)   
    print(f'auprc: {mean(scores)} ± {stdev(scores)}')
    
    file = open('../results/preprocess_best/' + preprocess_method + '_eln_model_test_scores.save', 'wb')
    pickle.dump(scores, file)
    file.close()
    
    file = open('../results/preprocess_best/' + preprocess_method + '_eln_model_test_sets.save', 'wb')
    pickle.dump(final_test, file)
    file.close()

    file = open('../results/preprocess_best/' + preprocess_method + '_eln_model_test_models.save', 'wb')
    pickle.dump(final_models, file)
    file.close()