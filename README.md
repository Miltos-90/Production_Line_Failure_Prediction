# Prediction of internal failures in a production line

The dataset comes from the [Bosch production line performance competition](https://www.kaggle.com/c/bosch-production-line-performance/), in which we need to predict internal failures using thousands of measurements and tests made for each component along the assembly line. 

The data for this competition represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).

The dataset contains an extremely large number of anonymized features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.

## Libraries


```python
import utils
import metric

import numpy as np
import gc
import xgboost as xgb

import pickle
from bayes_opt import BayesianOptimization
from functools import partial
```

Let's have a look at the contents of the zipped files:

## Modelling


```python
# Load checkpoint (saved at the end of the EDA notebook)
file_name = "./datasets.pkl"
open_file = open(file_name, "rb")
X_train, X_holdout, y_train, y_holdout, skf = pickle.load(open_file)
open_file.close()
```

### Evaluation metric

We need a function to compute the Matthews Correlation Coefficient (MCC) in an efficient way for xgboost. We'll use some numba magic for this, so as to optimise the threshold probability as well:


```python
y_prob0 = np.random.rand(1000000)
y_prob  = y_prob0 + 0.4 * np.random.rand(1000000) - 0.02
y_true  = (y_prob0 > 0.6).astype(int)

%timeit metric.eval_mcc(y_true, y_prob)

del y_prob0, y_prob, y_true
gc.collect();
```

    168 ms ± 16.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

### k-fold CV

We'll use xgboost as the learning algorithm. Let's write a wrapper to perform k-fold CV and return the average validation MCC:


```python
# Make parameter set for Tree booster
params = {
    "eta": (0.05, 0.3), 
    "gamma": (0, 100),
    "max_depth": (5, 50), 
    "num_boost_round": (10, 100), 
    "subsample": (0.5, 0.95), 
    "colsample_bytree": (0.5, 0.95), 
    "alpha": (0, 10), 
    "lamda": (0, 10)} 

# Function handle
f = partial(utils.CV, X_train, y_train, skf)

optimizer = BayesianOptimization(f, params, random_state = 111)
optimizer.maximize(init_points = 20, n_iter = 10)

Let's train the best model on all the data:


```python
# Make dmatrices
dtrain = xgb.DMatrix(X_train, y_train)
dheld  = xgb.DMatrix(X_holdout, y_holdout.to_numpy())

# Scale positive instances
sum_neg, sum_pos = np.sum(y_train == 0), np.sum(y_train == 1)

# Make parameter dict for xgboost
xgb_params = {"nthread": -1, "booster":"gbtree", "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
              "eta":              optimizer.max["params"]["eta"], 
              "gamma":            optimizer.max["params"]["gamma"], 
              "max_depth":        int(optimizer.max["params"]["max_depth"]), 
              "subsample":        optimizer.max["params"]["subsample"],
              "alpha":            optimizer.max["params"]["alpha"], 
              "lambda":           optimizer.max["params"]["lamda"],
              "colsample_bytree": optimizer.max["params"]["colsample_bytree"],
             "scale_pos_weight" : sum_neg / sum_pos}

# Train using the parameters
clf = xgb.train(params = xgb_params,
                dtrain = dtrain,
                feval  = metric.mcc_eval,
                evals  = [(dtrain, 'train')],
                maximize = True,
                verbose_eval = False,
                num_boost_round = int(optimizer.max["params"]["num_boost_round"]),
                early_stopping_rounds = 10)
```

Let's predict on the heldout set and compute the MCC:


```python
y_prob = clf.predict(dheld)
print(f"Heldout Set MCC: {round(metric.eval_mcc(y_holdout.to_numpy(), y_prob), 3)}")
```

    Heldout Set MCC: 0.255
    
