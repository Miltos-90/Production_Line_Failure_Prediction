import numba
import numpy as np
from numba import jit

@jit(nopython = True)
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)

@jit(nopython = True)
def eval_mcc(y_true, y_prob):
    
    # Sort arrays
    idx         = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    
    # Get # samples
    n = y_true.shape[0]
    
    # Get no. positive and no. negative
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    
    best_mcc   = 0.0
    best_id    = -1
    prev_proba = -1
    best_proba = -1
    
    # Initialise array for results
    mccs = np.zeros(n)
    
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        
        if proba != prev_proba:
            
            prev_proba = proba
            new_mcc    = mcc(tp, tn, fp, fn)
            
            if new_mcc >= best_mcc:
                
                best_mcc   = new_mcc
                best_id    = i
                best_proba = proba
                
        mccs[i] = new_mcc
        
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    
    return best_mcc

# Evaluation metric compatible with xgboost api
def mcc_eval(y_prob, dtrain):
    
    y_true   = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    
    return 'MCC', best_mcc