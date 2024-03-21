#import libraries
import pandas as pd
import xgboost as xgb
import numpy as np

#import dataset
dataset = pd.read_csv("bank-full.csv", sep = ";")
dataset.dtypes

#isolate the x and y variabbles
y = dataset.iloc[:, -1].values
x = dataset._get_numeric_data()

#split data into training and test set
from sklearn.model_selection import train_test_split
#random_state variable is just my birthday
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size = 0.2, random_state = 1508)
    
#convert y variable into dummy variable
#currently y is yes and no, we should change it to 1s and 0s
y_train = np.where(y_train == "yes", 1, 0)
y_test = np.where(y_test == "yes", 1, 0)
np.mean(y_train) #12% of data is yeses

#create XGBoost matrices
train = xgb.DMatrix(x_train, label = y_train)
test = xgb.DMatrix(x_test, label = y_test)

#set parameters
parameters1 = {'learning_rate': 0.3,
               'max_depth': 2,
               'colsample_bytree': 1,
               'subsample': 1,
               'min_child_weight': 1,
               'gamma': 0,
               'random_state': 1508,
               'eval_metric': 'auc',
               'objective': 'binary:logistic'}

#Run XGBoost!
model1 = xgb.train(params = parameters1, 
                   dtrain = train,
                   num_boost_round = 200,
                   evals = [(test, "Yes")],
                   verbose_eval = 50)


#Predictions
predictions1 = model1.predict(test)
predictions1 = np.where(predictions1 > 0.5, 1, 0)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_matrix1 = confusion_matrix(y_test, predictions1)
print(confusion_matrix1)

report1 = classification_report(y_test, predictions1)
print(report1)

###########################################################################

#isolate categorical variables
dataset_categorical = dataset.select_dtypes(exclude = "number")

#transform categorical variables into dummy variables
dataset_categorical = pd.get_dummies(data = dataset_categorical, 
                                     drop_first = True)

#joining numerical and categorical datasets
final_dataset = pd.concat([x, dataset_categorical], axis = 1)

#getting names of columns
feature_columns = list(final_dataset.columns.values)
feature_columns = feature_columns[:-1]

############################################################################

#isolate the x and y variabbles part 2
y = final_dataset.iloc[:, -1].values
x = final_dataset.iloc[:, :-1].values

#random_state variable is just my birthday
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size = 0.2, random_state = 1508)

#create XGBoost matrices part 2
train = xgb.DMatrix(x_train, label = y_train, feature_names=feature_columns)
test = xgb.DMatrix(x_test, label = y_test, feature_names=feature_columns)

#set parameters part 2
parameters2 = {'learning_rate': 0.3,
               'max_depth': 2,
               'colsample_bytree': 1,
               'subsample': 1,
               'min_child_weight': 1,
               'gamma': 0,
               'random_state': 1508,
               'eval_metric': 'auc',
               'objective': 'binary:logistic'}

#Run XGBoost!
model2 = xgb.train(params = parameters2, 
                   dtrain = train,
                   num_boost_round = 200,
                   evals = [(test, "Yes")],
                   verbose_eval = 50)

#Predictions part 2
predictions2 = model2.predict(test)
predictions2 = np.where(predictions2 > 0.5, 1, 0)

#Confusion Matrix
confusion_matrix2 = confusion_matrix(y_test, predictions2)
print(confusion_matrix2)

report2 = classification_report(y_test, predictions2)
print(report2)

#####################################################################

#checking how many cpus are in this computer/google
import multiprocessing
multiprocessing.cpu_count()

#setting cross validation parameters
from sklearn.model_selection import KFold
#n_split should be 10 in practice, but this is an example so I don't wanna waste time
tune_control = KFold(n_splits = 5,
                     shuffle = True,
                     random_state=1508).split(X = x_train, y = y_train)

#set parameter tuning
#set parameters part 3
tune_grid = {'learning_rate': [0.05, 0.3],
               'max_depth': range(2,9,2),
               'colsample_bytree': [0.5, 1],
               'subsample': [1],
               'min_child_weight': [1],
               'gamma': [0],
               'random_state': [1508],
               'n_estimators': range(200, 2000, 200),
               'booster': ['gbtree']}

#State we are doing a classification problem
from xgboost import XGBClassifier
classifier = XGBClassifier(objective="binary:logistic")


#cross validation assembly
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, param_grid = tune_grid,
                           scoring = "roc_auc", n_jobs=2, #number of cpus
                           cv=tune_control,
                           verbose=5)

#Setting evaluation parameters
evaluation_parameters = {"early_stopping_rounds": 100,
                         "eval_metric": "auc",
                         "eval_set": [(x_test, y_test)]}


#Hyper parameter tuning and cross validation THIS STEP TAKES 2 HOURS ON 6 CPU
tune_model = grid_search.fit(X = x_train, y = y_train, **evaluation_parameters)
grid_search.best_params_, grid_search.best_score_

##############################################################################


#setting cross validation parameters
from sklearn.model_selection import KFold
#n_split should be 10 in practice, but this is an example so I don't wanna waste time
tune_control = KFold(n_splits = 5,
                     shuffle = True,
                     random_state=1508).split(X = x_train, y = y_train)

#set parameter tuning part 2
tune_grid2 = {'learning_rate': [0.05],
               'max_depth': [6],
               'colsample_bytree': [0.5],
               'subsample': [0.9, 1],
               'min_child_weight': range(0,5,1),
               'gamma': [0, 0.1],
               'random_state': [1508],
               'n_estimators': range(200, 2000, 200),
               'booster': ['gbtree']}


#cross validation assembly
from sklearn.model_selection import GridSearchCV
grid_search2 = GridSearchCV(estimator = classifier, param_grid = tune_grid2,
                           scoring = "roc_auc", n_jobs=2, #number of cpus
                           cv=tune_control,
                           verbose=5)


#Hyper parameter tuning and cross validation THIS STEP TAKES 2 HOURS ON 6 CPU
tune_model2 = grid_search2.fit(X = x_train, y = y_train, **evaluation_parameters)
grid_search2.best_params_, grid_search2.best_score_


###############################################################################

#set parameters part 3
parameters3 = {'learning_rate': 0.01,
               'max_depth': 14,
               'colsample_bytree': 0.5,
               'subsample': 0.4,
               'min_child_weight': 1,
               'gamma': 0.1,
               'random_state': 1508,
               'eval_metric': 'auc',
               'objective': 'binary:logistic'}

#Run XGBoost!
model3 = xgb.train(params = parameters3, 
                   dtrain = train,
                   num_boost_round = 800,
                   evals = [(test, "Yes")],
                   verbose_eval = 50)

#Predictions part 3
predictions3 = model3.predict(test)
predictions3 = np.where(predictions3 > 0.1, 1, 0)

#Confusion Matrix
confusion_matrix3 = confusion_matrix(y_test, predictions3)
print(confusion_matrix3)

report3 = classification_report(y_test, predictions3)
print(report3)


# Should continue tuning data