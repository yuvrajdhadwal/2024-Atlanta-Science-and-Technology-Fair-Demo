import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report

dataset = pd.read_csv("test-data_000(without_timestamps).csv")
poses = dataset.iloc[:,0]
points = dataset.iloc[:,1:]

points_train, points_test, poses_train, poses_test = \
    train_test_split(points, poses, test_size = 0.2, random_state = 1508)
    
train = xgb.DMatrix(points_train, label = poses_train)
test = xgb.DMatrix(points_test, label = poses_test)

#random right now
parameters = {'learning_rate': 0.3,
               'max_depth': 2,
               'num_class': 19,
               'colsample_bytree': 1,
               'subsample': 1,
               'min_child_weight': 1,
               'gamma': 0,
               'random_state': 1508,
               'eval_metric': 'map@3',
               'objective': 'multi:softproba'}

clf = xgb.XGBClassifier(**parameters, tree_method="exact", booster="dart")

clf.fit(points_train, poses_train)

test_df = pd.concat([poses_test, points_test], axis=1)
y_pred = clf.predict_proba(points_test)
y_pred = pd.DataFrame(y_pred)
y_pred

def top_3(arr): 
    n = arr.shape[0]
    out = np.zeros((n, 3))
    for i in range(0, n):
        out[i, ] = arr[i,].argsort()[::-1][:3]
    return out


top_3_prediction_test = top_3(clf.predict_proba(points_test)).astype('int32')
print(top_3_prediction_test) 
poses_test