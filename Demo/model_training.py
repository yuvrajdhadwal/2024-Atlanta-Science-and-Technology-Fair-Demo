import pandas as pd
import numpy as np
import xgboost as xgb

"""
This script uses XGBoost to train model based on training data and pre-tuned parameters
Then saves this model for use in actual exhibit.

Yuvraj Dhadwal 2/24/2024
"""


dataframe = pd.read_csv("data/merged_data.csv")
labels = dataframe.iloc[:, 2:4]
features = dataframe.iloc[:, 5:]
#features.to_excel("features.xlsx")
train = xgb.DMatrix(features, label=labels)

# currently random numbers needs to be tuned based on data we get
parameters = {
    "learning_rate": 0.3,
    "max_depth": 2,
    "colsample_bytree": 1,
    "subsample": 1,
    "min_child_weight": 1,
    "gamma": 0,
    "random_state": 1508,
    "eval_metric": "map@3",
}

model = xgb.XGBRegressor(**parameters, tree_method="hist", booster="dart")

model.fit(features, labels)

model.save_model("pose_recognition_model.bin")