import pandas as pd
from sklearn.utils import shuffle

def binarize_data(df):
    num_cols = list(df.select_dtypes(exclude=['O']).columns)
    df.loc[:,num_cols] = (df.loc[:,num_cols] > 0.0).astype(int)
    return df

def customized_cv_index(train):
    index_13, index_24 = train.loc[(train['animal'] == 7)|(train['animal'] == 3),].index, train.loc[(train['animal'] == 8)|(train['animal'] == 4),].index
    
    index_14, index_23 = train.loc[(train['animal'] == 7) | (train['animal'] == 4),].index, train.loc[(train['animal'] == 8)|(train['animal'] == 3),].index
    
    custom_cv = [(index_13, index_24), 
                 (index_14, index_23),
                 (index_23, index_14),
                 (index_24, index_13)]
    return custom_cv

def data_prep(input_test, input_train, cell_type, binarization=True):
    '''
    Prepare training and testing sets for group-based cross-validation.
    
    Training and testing uses the same cell type.
    '''
    
    df_test = pd.read_csv(input_test, index_col=0)
    df_train = pd.read_csv(input_train, index_col=0)

    if (cell_type=='All'):
        test_idx = df_test.index
        train_idx = df_train.index        
        
    elif (cell_type=='Non-neuronal'):
        test_idx= df_test.loc[df_test.major_group!='Neuron'].index      
        train_idx= df_train.loc[df_train.major_group!='Neuron'].index                          
                             
    else: 
        test_idx = df_test.loc[df_test.major_group==cell_type].index
        train_idx = df_train.loc[df_train.major_group==cell_type].index

    assert len(test_idx)>0, "This cell type doesn't exit in the test set. \n Or you may have a typo :("
    assert len(train_idx)>0, "This cell type doesn't exit in the training set. \n Or you may have a typo :("

    test_X = df_test.loc[test_idx].iloc[:,:-3]
    test_y = df_test.target
    test_y = test_y.loc[test_idx]
    test_X, test_y = shuffle(test_X, test_y, random_state=42)

    df_train = df_train.loc[train_idx]
    train = df_train.reset_index()

    custom_cv = customized_cv_index(train)
    
    train_X = train.iloc[:,1:-3]
    train_y = train.target
    
    if binarization==True:
        test_X = binarize_data(test_X)
        train_X = binarize_data(train_X)
    
    print('Finished data prepration for ' + str(cell_type) )
    return train_X, train_y, test_X, test_y, custom_cv

def data_perp_cell_type(input_test, input_train, train_cell_type, test_cell_type, binarization=True):
    '''
    Prepare training and testing sets for group-based cross-validation.
    
    Training and testing may use different cell types.
    '''
    
    df_train = pd.read_csv(input_train, index_col=0)
    if (train_cell_type=='All'):
        train_idx = df_train.index        
        
    elif (train_cell_type=='Non-neuronal'):
        train_idx= df_train.loc[df_train.major_group!='Neuron'].index                          
                             
    else: 
        train_idx = df_train.loc[df_train.major_group==train_cell_type].index
    
    df_test = pd.read_csv(input_test, index_col=0)
    if (test_cell_type=='All'):
        test_idx = df_test.index        
        
    elif (test_cell_type=='Non-neuronal'):
        test_idx= df_test.loc[df_test.major_group!='Neuron'].index                          
                             
    else: 
        test_idx = df_test.loc[df_test.major_group==test_cell_type].index
    
    
    assert len(test_idx)>0, "This cell type doesn't exit in the test set. \n Or you may have a typo :("
    assert len(train_idx)>0, "This cell type doesn't exit in the training set. \n Or you may have a typo :("

    test_X = df_test.loc[test_idx].iloc[:,:-3]
    test_y = df_test.target
    test_y = test_y.loc[test_idx]
    test_X, test_y = shuffle(test_X, test_y, random_state=42)
    

    df_train = df_train.loc[train_idx]
    train = df_train.reset_index()
    
    custom_cv = customized_cv_index(train)

    train_X = train.iloc[:,1:-3]
    train_y = train['target']
    
    if binarization==True:
        test_X = binarize_data(test_X)
        train_X = binarize_data(train_X)
    
    print('Finished data prepration for ' + str(train_cell_type) + 
          ' in the training set and ' + str(test_cell_type) + 
         ' in the testing set')
    return train_X, train_y, test_X, test_y, custom_cv