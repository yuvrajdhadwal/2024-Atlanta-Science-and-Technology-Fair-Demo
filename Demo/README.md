# Read Me

This is the main folder for the demo. It collects data and then trains a model based on that data to detect motion change from a camera. There is also a file that will run and display the fully functional model predicting your movements.

Run `model_training.py` to run the script that was displayed during my presentation at the Atlanta Science and Technology Fair. This model should display two windows. On the right, a window will appear that displays the footage the camera is capturing alongside the points of the body that the program is tracking. On the left, a window will appear with two dots. The red dot is the training program we used to collect data to train the model and the blue dot is the Machine Learning prediction point of your entire body. As you move in the real world, the blue dot should move alongside you regardless of whether you jump, squat, walk left/right, or even dance!

# How it works:

### Step 1:

Run `run.py` to start both files. Press `q` to quit the windows (this may be buggy).

This file will run `data_collection.py` and `video.py` simultaneously.

`data_collection.py` is a script that will take all the points of a person's body and attaches it to a timestamp and output a CSV file of the contents. These are known as the features of the Machine Learning Model.

`video.py` is a script that will display a screen that has a red dot that bounces around the screen. This script will also output a CSV with a timestamp and the (x,y) cartesian coordinate of the red dot. These are known as the values of the Machine Learning Model.

Now, if you were to run this file with a person in front of the camera and instruct them to follow the red dot with their body, you would be able to have a good sense of where each point of the body is for each (x,y) coordinate in the window if you were to merge the data by the timestamp.

### Step 2:

Redo Step 1 until you have collected sufficient data. For this presentation, I redid Step 1 three times per person for seven different people. This resulted in 21 different files for our features and values.

### Step 3:

Run `data_merging.py` to automatically match the data collected from `data_collection.py` and `video.py` for each set of data you collect based on timestamp. Then, it will concatenate all the data you collected into a single Excel file called `merged_data.csv`. For this demo, we were able to gather 650,000 data points. This is a far cry more than the 1000 data points collected last semester with a different approach. Both approaches took around 90 minutes in the afternoon to run and collect data.

### Step 4:

This would be the step to tune the parameters of the final Machine Learning Model based on the data collected by creating Testing and Training Datasets. However, since I was short on time for this project because the deadline was coming up this step was skipped. The Machine Learning Model was reasonably accurate in its predictions and good enough for the presentation.

### Step 5:

Update the hyperparameters in `model_training.py` to create the predictive XGBoost model based on your results from Step 4 for your dataset.

Run `model_training.py` to output the Machine Learning model as `pose_recognition_model.bin` for use from other files and for later use so we do not need to keep creating a new model every single time we run this program.

Note: The parameters used in this file are copied from the Learning Folder parameters these are not tuned to this dataset and are not optimized.

### Step 6:

Use the model exported from this last file in whatever file you want! You can also run `model_testing.py` for to see how good the model is at prediction. Read the top of this file to see what `model_testing.py` does.
