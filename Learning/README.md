# MLModels-Bank-Full.py
Based on this course: https://www.udemy.com/course/xgboost-python-r/?couponCode=ST14MT32124

I created a Python file that creates a Machine Learning Model to fit the data bank-full.csv. Based on the given features, the model predicts if the Bank should offer a loan to an individual. This model is given a dataset and then it splits up the dataset randomly to create a training dataset and a testing dataset. Then the program tunes the data parameters to create a more effective Machine Learning Model on it. It prints out the Confusion Matrices to prove that the tuning is effective. This is a binary classification Machine Learning model. 

# MLModels-Bank-Full-GPU.py

Annoyed with the fact that tuning the model parameters took around 2 hours running on 6 CPU cores, I set out to tune the data a faster way by utilizing the GPU. By utilizing the GPU I was able to shrink the time required to tune the model parameters based on the data to only 3 minutes. This decrease in time will prove vital when the data is much larger than 45,000 lines on an Excel file.

# multi_softproba_classification.py

One issue with the previous models is that they were binary classifications. Meaning the model would spit out either a Yes or No answer. If we are trying to pick a point on the screen we will need hundreds of points to classify. Right now, we only have training data for 19 different points, so let us try at least to classify between those 19 points. This program achieves that. For the actual demo, I will need to come up with a robust way to predict points based on points on the body rather than classifying body poses based on points on the body.
