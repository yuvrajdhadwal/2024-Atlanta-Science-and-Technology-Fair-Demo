import pandas as pd
import xgboost as xgboost
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

#Calculating how long the fucntion takes to run
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#reading data and converting it into dataset
dataset = pd.read_csv("bank-full.csv", sep = ";")
dataset_numerical = dataset._get_numeric_data()
dataset_categorical = dataset.select_dtypes(exclude="number")
dataset_categorical = pd.get_dummies(data = dataset_categorical, drop_first=True)
dataset_final = pd.concat([dataset_numerical, dataset_categorical], axis = 1)

#create training
y = dataset_final.iloc[:, -1].values
X = dataset_final.iloc[:, :-1].values

x_train, x_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.2, random_state = 1508)
    
params = {
        'min_child_weight': [1],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
        'subsample': [0.4, 0.8, 1.0, 1.2, 1.5],
        'colsample_bytree': [0.1, 0.25, 0.5, 1.0],
        'max_depth': range(2,16,2),
        'learning_rate': [0.05]    
        }

folds = 5
param_comb = 1

target = 'target'

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1508)

xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=6, tree_method='hist', device = 'cuda', eval_metric='auc')
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X=x_train,y=y_train), verbose=3, random_state=1508)

#Setting evaluation parameters
evaluation_parameters = {"eval_set": [(x_test, y_test)]}


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
#Hyper parameter tuning and cross validation THIS STEP TAKES 2 HOURS ON 6 CPU
tune_model = random_search.fit(X = x_train, y = y_train, **evaluation_parameters)
print(random_search.best_params_, random_search.best_score_)
timer(start_time) # timing ends here for "start_time" variable

