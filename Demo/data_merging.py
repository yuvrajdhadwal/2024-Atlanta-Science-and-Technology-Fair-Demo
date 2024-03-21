import pandas as pd
import glob

"""
This script reads data from csv files created from video.py and data_collection.py.
It then merges it into one dataframe and then saves it into another dataframe that
is a collection of all the data into one single excel file. This excel file will then
be exported for use by XGBoost.

Yuvraj Dhadwal 2/25/2024
"""


excel_number = 0
all_data = pd.DataFrame()


# excel_number should be given by glob to find the latest file, then increment the number by 1

files = glob.glob("data/*.*")
if len(files) == 0:
    raise Exception("No files found")
else:
    num = len(files) / 2
print(len(files))

while excel_number < num:
    excel_number = excel_number + 1

    features = pd.read_csv("data/training_values_{0:03}.csv".format(excel_number))
    labels = pd.read_csv("data/point_coordinates_{0:03}.csv".format(excel_number))

    if "Time" not in features.columns or "Time" not in labels.columns:
        raise ValueError("One of the DataFrames does not have the 'Time' column")

    features["Hip Width"] = abs(features["x_rightHip"] - features["x_leftHip"])
    features["Torso Height"] = abs(features["y_leftEye"] - features["y_leftHip"])
    features["Shoulder Width"] = abs(
        features["x_leftShoulder"] - features["x_rightShoulder"]
    )
    features["Leg Length"] = abs(features["y_leftFoot"] - features["y_leftHip"])
    features["Height"] = abs(features["y_leftEye"] - features["y_leftFoot"])
    features["Height to Hips Ratio"] = abs(features["Height"] / features["Hip Width"])
    # features = features.iloc[:, :]

    merged_features = pd.merge(labels, features, on="Time")
    merged_features.reset_index(drop=True, inplace=True)

    all_data = pd.concat([all_data, merged_features])

all_data.to_csv("data/merged_data.csv")
