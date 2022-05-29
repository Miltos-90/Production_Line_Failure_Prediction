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

```

    |   iter    |  target   |   alpha   | colsam... |    eta    |   gamma   |   lamda   | max_depth | num_bo... | subsample |
    -------------------------------------------------------------------------------------------------------------------------
    | [0m 1       [0m | [0m 0.2763  [0m | [0m 6.122   [0m | [0m 0.5761  [0m | [0m 0.159   [0m | [0m 76.93   [0m | [0m 2.953   [0m | [0m 11.71   [0m | [0m 12.02   [0m | [0m 0.6891  [0m |
    | [95m 2       [0m | [95m 0.2893  [0m | [95m 2.387   [0m | [95m 0.6519  [0m | [95m 0.2977  [0m | [95m 23.77   [0m | [95m 0.8119  [0m | [95m 35.13   [0m | [95m 65.91   [0m | [95m 0.6234  [0m |
    | [95m 3       [0m | [95m 0.3089  [0m | [95m 4.662   [0m | [95m 0.5533  [0m | [95m 0.06849 [0m | [95m 90.08   [0m | [95m 7.94    [0m | [95m 42.83   [0m | [95m 83.37   [0m | [95m 0.9459  [0m |
    | [0m 4       [0m | [0m 0.2689  [0m | [0m 5.773   [0m | [0m 0.8662  [0m | [0m 0.1553  [0m | [0m 2.745   [0m | [0m 4.541   [0m | [0m 9.74    [0m | [0m 83.55   [0m | [0m 0.814   [0m |
    | [95m 5       [0m | [95m 0.3226  [0m | [95m 5.653   [0m | [95m 0.6234  [0m | [95m 0.2996  [0m | [95m 13.8    [0m | [95m 6.154   [0m | [95m 26.83   [0m | [95m 46.45   [0m | [95m 0.8275  [0m |
    | [0m 6       [0m | [0m 0.2902  [0m | [0m 3.228   [0m | [0m 0.6802  [0m | [0m 0.1297  [0m | [0m 94.72   [0m | [0m 9.187   [0m | [0m 41.62   [0m | [0m 13.07   [0m | [0m 0.9245  [0m |
    | [0m 7       [0m | [0m 0.262   [0m | [0m 9.504   [0m | [0m 0.863   [0m | [0m 0.1703  [0m | [0m 96.68   [0m | [0m 4.17    [0m | [0m 19.35   [0m | [0m 11.48   [0m | [0m 0.5172  [0m |
    | [0m 8       [0m | [0m 0.2773  [0m | [0m 0.5309  [0m | [0m 0.5576  [0m | [0m 0.05835 [0m | [0m 22.64   [0m | [0m 5.439   [0m | [0m 13.04   [0m | [0m 26.14   [0m | [0m 0.5673  [0m |
    | [0m 9       [0m | [0m 0.3113  [0m | [0m 6.83    [0m | [0m 0.7111  [0m | [0m 0.2421  [0m | [0m 66.88   [0m | [0m 4.413   [0m | [0m 31.79   [0m | [0m 51.55   [0m | [0m 0.9335  [0m |
    | [0m 10      [0m | [0m 0.2662  [0m | [0m 1.478   [0m | [0m 0.9081  [0m | [0m 0.1745  [0m | [0m 59.25   [0m | [0m 6.847   [0m | [0m 12.99   [0m | [0m 87.82   [0m | [0m 0.7169  [0m |
    | [0m 11      [0m | [0m 0.2749  [0m | [0m 0.6934  [0m | [0m 0.6641  [0m | [0m 0.1404  [0m | [0m 47.21   [0m | [0m 1.795   [0m | [0m 12.89   [0m | [0m 95.42   [0m | [0m 0.7951  [0m |
    | [0m 12      [0m | [0m 0.2736  [0m | [0m 1.35    [0m | [0m 0.9175  [0m | [0m 0.2604  [0m | [0m 5.278   [0m | [0m 2.037   [0m | [0m 44.26   [0m | [0m 70.65   [0m | [0m 0.6338  [0m |
    | [0m 13      [0m | [0m 0.2941  [0m | [0m 7.344   [0m | [0m 0.7783  [0m | [0m 0.1026  [0m | [0m 80.21   [0m | [0m 0.5911  [0m | [0m 18.87   [0m | [0m 28.66   [0m | [0m 0.9375  [0m |
    | [0m 14      [0m | [0m 0.2784  [0m | [0m 9.094   [0m | [0m 0.8044  [0m | [0m 0.0788  [0m | [0m 95.8    [0m | [0m 0.9171  [0m | [0m 33.34   [0m | [0m 91.26   [0m | [0m 0.6529  [0m |
    | [0m 15      [0m | [0m 0.2653  [0m | [0m 6.528   [0m | [0m 0.9163  [0m | [0m 0.1435  [0m | [0m 77.14   [0m | [0m 8.962   [0m | [0m 32.91   [0m | [0m 23.6    [0m | [0m 0.6687  [0m |
    | [0m 16      [0m | [0m 0.2946  [0m | [0m 6.358   [0m | [0m 0.7066  [0m | [0m 0.1831  [0m | [0m 35.23   [0m | [0m 6.599   [0m | [0m 26.44   [0m | [0m 50.91   [0m | [0m 0.5562  [0m |
    | [0m 17      [0m | [0m 0.289   [0m | [0m 4.125   [0m | [0m 0.6229  [0m | [0m 0.1367  [0m | [0m 97.44   [0m | [0m 2.156   [0m | [0m 35.73   [0m | [0m 12.86   [0m | [0m 0.8759  [0m |
    | [0m 18      [0m | [0m 0.27    [0m | [0m 7.017   [0m | [0m 0.9347  [0m | [0m 0.198   [0m | [0m 21.73   [0m | [0m 7.192   [0m | [0m 6.389   [0m | [0m 47.68   [0m | [0m 0.6266  [0m |
    | [0m 19      [0m | [0m 0.2975  [0m | [0m 1.062   [0m | [0m 0.6919  [0m | [0m 0.2601  [0m | [0m 11.04   [0m | [0m 9.867   [0m | [0m 15.5    [0m | [0m 74.93   [0m | [0m 0.7974  [0m |
    | [0m 20      [0m | [0m 0.2833  [0m | [0m 0.7776  [0m | [0m 0.8138  [0m | [0m 0.1163  [0m | [0m 90.18   [0m | [0m 8.183   [0m | [0m 22.97   [0m | [0m 83.83   [0m | [0m 0.589   [0m |
    | [95m 21      [0m | [95m 0.3334  [0m | [95m 6.459   [0m | [95m 0.5462  [0m | [95m 0.2506  [0m | [95m 36.25   [0m | [95m 6.777   [0m | [95m 25.78   [0m | [95m 49.77   [0m | [95m 0.9422  [0m |
    | [0m 22      [0m | [0m 0.3012  [0m | [0m 4.741   [0m | [0m 0.7108  [0m | [0m 0.1743  [0m | [0m 13.35   [0m | [0m 7.41    [0m | [0m 26.38   [0m | [0m 48.17   [0m | [0m 0.7132  [0m |
    | [0m 23      [0m | [0m 0.2815  [0m | [0m 3.441   [0m | [0m 0.8176  [0m | [0m 0.1366  [0m | [0m 14.58   [0m | [0m 4.772   [0m | [0m 25.93   [0m | [0m 46.8    [0m | [0m 0.5528  [0m |
    | [0m 24      [0m | [0m 0.279   [0m | [0m 3.108   [0m | [0m 0.8705  [0m | [0m 0.1189  [0m | [0m 87.29   [0m | [0m 8.282   [0m | [0m 41.84   [0m | [0m 83.65   [0m | [0m 0.6098  [0m |
    | [0m 25      [0m | [0m 0.2937  [0m | [0m 6.576   [0m | [0m 0.86    [0m | [0m 0.2396  [0m | [0m 34.55   [0m | [0m 6.366   [0m | [0m 25.5    [0m | [0m 50.53   [0m | [0m 0.7603  [0m |
    | [0m 26      [0m | [0m 0.2701  [0m | [0m 5.059   [0m | [0m 0.8118  [0m | [0m 0.09802 [0m | [0m 36.69   [0m | [0m 8.31    [0m | [0m 26.07   [0m | [0m 51.84   [0m | [0m 0.6583  [0m |
    | [0m 27      [0m | [0m 0.2943  [0m | [0m 4.404   [0m | [0m 0.6211  [0m | [0m 0.2609  [0m | [0m 36.23   [0m | [0m 7.353   [0m | [0m 23.56   [0m | [0m 47.98   [0m | [0m 0.667   [0m |
    | [0m 28      [0m | [0m 0.2895  [0m | [0m 1.508   [0m | [0m 0.6033  [0m | [0m 0.1134  [0m | [0m 12.35   [0m | [0m 9.101   [0m | [0m 16.4    [0m | [0m 75.78   [0m | [0m 0.7704  [0m |
    | [0m 29      [0m | [0m 0.3153  [0m | [0m 5.752   [0m | [0m 0.6329  [0m | [0m 0.1538  [0m | [0m 14.05   [0m | [0m 7.219   [0m | [0m 28.18   [0m | [0m 45.32   [0m | [0m 0.7804  [0m |
    | [0m 30      [0m | [0m 0.2692  [0m | [0m 5.248   [0m | [0m 0.9104  [0m | [0m 0.1756  [0m | [0m 15.03   [0m | [0m 5.825   [0m | [0m 26.13   [0m | [0m 49.58   [0m | [0m 0.5196  [0m |
    =========================================================================================================================
    

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
    
