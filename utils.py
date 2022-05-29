import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import xgboost as xgb
import metric

def extract_line(colname):    return colname.split('_')[0]
def extract_station(colname): return colname.split('_')[1]
def extract_feature(colname): return colname.split('_')[2]

def CV(X_train, y_train, skf, eta, gamma, max_depth, num_boost_round, subsample, colsample_bytree, alpha, lamda):
    
    # Early stopping
    early_stopping_rounds = 10

    # Make parameter dict for xgboost
    xgb_params = {"nthread": -1, "booster":"gbtree", "objective": "binary:logistic", 
                  "eval_metric": "auc", "tree_method": "hist", "eta": eta, "gamma": gamma, 
                  "max_depth": int(max_depth), "subsample": subsample,
                  "colsample_bytree": colsample_bytree, "alpha": alpha, "lambda": lamda}
    
    # Run CV with the given parameters
    scores = []
    for train_idx, val_idx in skf.split(np.zeros(y_train.shape[0]), y_train):

        # Make train and validation sets for this fold
        X_train_, y_train_ = X_train[train_idx, :], y_train[train_idx]
        X_val_, y_val_     = X_train[val_idx, :], y_train[val_idx]

        # Make dmatrix
        dtrain = xgb.DMatrix(X_train_, y_train_)
        dval   = xgb.DMatrix(X_val_, y_val_)

        # Scale positive instances
        sum_neg, sum_pos = np.sum(y_train_ == 0), np.sum(y_train_ == 1)
        xgb_params["scale_pos_weight"] = sum_neg / sum_pos

        # Train using the parameters
        bst = xgb.train(params = xgb_params,
                        dtrain = dtrain,
                        feval  = metric.mcc_eval,
                        evals  = [ (dtrain, 'train'), (dval, 'eval') ],
                        maximize = True,
                        verbose_eval = False,
                        num_boost_round = int(num_boost_round),
                        early_stopping_rounds = early_stopping_rounds)

        # Grab the best score on the validation set
        scores.append(bst.best_score)
        
    return np.mean(scores)

# Smoothened mean value
def calc_smooth_mean(df, by, on, m):
    '''
    Credit: https://maxhalford.github.io/blog/target-encoding/
    '''
    
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg    = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means  = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)

# Compute station transition sequence from one record
def station_seq(date_df_row):
    
    # Sort by time and get index (original df column: line_station_feature)
    date_df_row = date_df_row.sort_values().dropna().to_frame().drop_duplicates().index
    
    # Make a sequence of stations from the index 
    seq = map(lambda x: str(x.split('_')[1].strip('S')), date_df_row)
    
    # Return unique while keeping order
    seq = list(dict.fromkeys(seq))
    
    return '_'.join(seq)

# Get diagonal and lower triangular pairs of correlation matrix
def _get_redundant_pairs(df):
    
    pairs_to_drop = set()
    cols          = df.columns
    
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    
    return pairs_to_drop

# Compute correlation matrix
def get_correlations(df):
    
    au_corr = df.corr().unstack()
    to_drop = _get_redundant_pairs(df)
    au_corr = au_corr.drop(labels = to_drop).sort_values(ascending = False).reset_index()
    au_corr.columns = ['feat1', 'feat2', 'corr']
    
    return au_corr

# Train/test split of sklearn takes quite a bit of time
# The following is faster to get a stratified train/test split
def train_test_split(y, test_ratio):
    
    # Grab indices of positive and negative responses
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Shuffle them
    np.random.seed(220)
    pos_idx = np.random.permutation(pos_idx)

    np.random.seed(220)
    neg_idx = np.random.permutation(neg_idx)

    # Split on train and test sets
    pos_samples = pos_idx.shape[0]
    neg_samples = neg_idx.shape[0]

    # Grab indices for the holdout set
    pos_idx_hold = pos_idx[:np.floor(pos_samples * test_ratio).astype(int)]
    neg_idx_hold = neg_idx[:np.floor(neg_samples * test_ratio).astype(int)]

    # Grab indices for the training set
    pos_idx_train = np.setdiff1d(pos_idx, pos_idx_hold)
    neg_idx_train = np.setdiff1d(neg_idx, neg_idx_hold)

    # Merge them
    idx_hold  = np.hstack([pos_idx_hold, neg_idx_hold])
    idx_train = np.hstack([pos_idx_train, neg_idx_train])
    
    # Shuffle once more
    np.random.seed(220)
    idx_hold = np.random.permutation(idx_hold)
    np.random.seed(220)
    idx_train = np.random.permutation(idx_train)
    
    return idx_train, idx_hold

# Function to load a dataframe from an input file
def load_df(input_file, d_type, fill_value, chunksize, no_rows):
    
    
    reader = pd.read_csv(input_file, chunksize = chunksize, engine = 'c', compression = 'zip', index_col = 'Id', dtype = d_type)

    dfs = []

    for df_chunk in tqdm(reader, total = no_rows // chunksize + 1):
        dfs.append(df_chunk.astype(pd.SparseDtype(d_type, fill_value = fill_value)))

    df = pd.concat(dfs, axis = 0)
    
    df.index = df.index.astype(int)

    return df