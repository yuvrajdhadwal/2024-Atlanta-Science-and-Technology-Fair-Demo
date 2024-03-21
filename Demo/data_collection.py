import cv2
import mediapipe as mp
import time
import pandas as pd
import glob
import os
import cv2
import numpy as np
import time

ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(
                frame, self.alpha, self.previous_frame, 1 - self.alpha, 0
            )
        self.previous_frame = result
        return result


"""
This script uses MediaPipe to detect points on your body and collects them as
numerical values and stores them in a XLSX file that the XGBoost program will read.

Make sure to rename the filename before running this script so you do not overwrite data.

Yuvraj Dhadwal 2/24/2024

'''
modelType = "Ducking"
folder_path = "./" + modelType + "./Features./"
filename = folder_path + "training_values_001.xlsx"  # Important note: please change
=======
"""

# use glob to find the latest file, then increment the number by 1

files = glob.glob("data/*.csv")
if len(files) == 0 or len(files) == 1:
    filename = "training_values_001.csv"
else:
    files.sort()
    filename = files[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    filename = filename[-1]
    filename = int(filename) + 1
    filename = "training_values_" + str(filename).zfill(3) + ".csv"


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


directory = "data"
# if the directory does not exist, then the program will create the directory

try:
    os.mkdir(directory)
except FileExistsError:
    pass


# calculates time for preferred format
def find_time():
    current_time = time.localtime(time.time())
    time_string = ":".join(
        "0%s" % (current_time[i]) if current_time[i] < 10 else "%s" % (current_time[i])
        for i in range(3, 6)
    )
    return time_string


fields = [
    "Time",
    "x_nose",
    "x_rightHip",
    "x_leftHip",
    "y_rightKnee",
    "y_leftKnee",
    "y_rightHip",
    "y_leftHip",
    "y_rightShoulder",
    "y_leftShoulder",
    "x_rightShoulder",
    "x_leftShoulder",
    "y_leftAnkle",
    "x_leftAnkle",
    "y_rightAnkle",
    "x_rightAnkle",
    "y_leftHeel",
    "x_leftHeel",
    "y_rightHeel",
    "x_leftHeel",
    "y_leftFoot",
    "x_leftFoot",
    "y_rightFoot",
    "x_rightFoot",
    "y_leftEye_inner",
    "x_leftEye_inner",
    "y_leftEye",
    "x_leftEye",
    "y_leftEye_outer",
    "x_leftEye_outer",
    "y_rightEye_inner",
    "x_rightEye_inner",
    "y_rightEye",
    "x_rightEye",
    "y_rightEye_outer",
    "x_rightEye_outer",
    "y_leftEar",
    "x_leftEar",
    "y_rightEar",
    "x_rightEar",
    "y_leftMouth",
    "x_leftMouth",
    "y_rightMouth",
    "x_rightMouth",
    "y_leftElbow",
    "x_leftElbow",
    "y_rightElbow",
    "x_rightElbow",
    "y_leftWrist",
    "x_leftWrist",
    "y_rightWrist",
    "x_rightWrist",
    "y_leftPinky",
    "x_leftPinky",
    "y_rightPinky",
    "x_rightPinky",
    "y_leftIndex",
    "x_leftIndex",
    "y_rightIndex",
    "x_rightIndex",
    "y_leftThumb",
    "x_leftThumb",
    "y_rightThumb",
    "x_rightThumb",
]
values = []  # default value for values,
# if the program is successful (see the try block below),
# then it should give the test values, otherwise it will print an empty string: " "
data_collection = {}


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        ## get height and width of the image
        height, width, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            continue
        data_collection.update({fields[0]: find_time()})
        x_nose = landmarks[mp_pose.PoseLandmark.NOSE].x
        data_collection.update({fields[1]: x_nose})
        y_nose = landmarks[mp_pose.PoseLandmark.NOSE].y
        x_rightHip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
        data_collection.update({fields[2]: x_rightHip})
        x_leftHip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
        data_collection.update({fields[3]: x_leftHip})
        y_rightKnee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
        data_collection.update({fields[4]: y_rightKnee})
        y_leftKnee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
        data_collection.update({fields[5]: y_leftKnee})
        y_rightHip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        data_collection.update({fields[6]: y_rightHip})
        y_leftHip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        data_collection.update({fields[7]: y_leftHip})

        # 11 - left shoulder & 12 - right shoulder
        y_rightShoulder = landmarks[12].y
        data_collection.update({fields[8]: y_rightShoulder})
        y_leftShoulder = landmarks[11].y
        data_collection.update({fields[9]: y_leftShoulder})
        x_rightShoulder = landmarks[12].x
        data_collection.update({fields[10]: x_rightShoulder})
        x_leftShoulder = landmarks[11].x
        data_collection.update({fields[11]: x_leftShoulder})
        # 27 - left ankle
        y_leftAnkle = landmarks[27].y
        data_collection.update({fields[12]: y_leftAnkle})
        x_leftAnkle = landmarks[27].x
        data_collection.update({fields[13]: x_leftAnkle})
        # 28 - right ankle
        y_rightAnkle = landmarks[28].y
        data_collection.update({fields[14]: y_rightAnkle})
        x_rightAnkle = landmarks[28].x
        data_collection.update({fields[15]: y_rightAnkle})
        # 29 - left heel
        y_leftHeel = landmarks[29].y
        data_collection.update({fields[16]: y_leftHeel})
        x_leftHeel = landmarks[29].x
        data_collection.update({fields[17]: x_leftHeel})
        # 30 - right heel
        y_rightHeel = landmarks[30].y
        data_collection.update({fields[18]: y_rightHeel})
        x_rightHeel = landmarks[30].x
        data_collection.update({fields[19]: y_leftHeel})
        # 31 - left foot index
        y_leftFoot = landmarks[31].y
        data_collection.update({fields[20]: y_leftFoot})
        x_leftFoot = landmarks[31].x
        data_collection.update({fields[21]: x_leftFoot})
        # 32 - right foot index
        y_rightFoot = landmarks[32].y
        data_collection.update({fields[22]: y_rightFoot})
        x_rightFoot = landmarks[32].x
        data_collection.update({fields[23]: x_rightFoot})
        # 1 - left eye (inner)
        y_leftEye_inner = landmarks[1].y
        data_collection.update({fields[24]: y_leftEye_inner})
        x_leftEye_inner = landmarks[1].x
        data_collection.update({fields[25]: x_leftEye_inner})
        # 2 - left eye
        y_leftEye = landmarks[2].y
        data_collection.update({fields[26]: y_leftEye})
        x_leftEye = landmarks[2].x
        data_collection.update({fields[27]: x_leftEye})
        # 3 - left eye (outer)
        y_leftEye_outer = landmarks[3].y
        data_collection.update({fields[28]: y_leftEye_outer})
        x_leftEye_outer = landmarks[3].x
        data_collection.update({fields[29]: x_leftEye_outer})
        # 4 - right eye (inner)
        y_rightEye_inner = landmarks[4].y
        data_collection.update({fields[30]: y_rightEye_inner})
        x_rightEye_inner = landmarks[4].x
        data_collection.update({fields[31]: x_rightEye_inner})
        # 5 - right eye
        y_rightEye = landmarks[5].y
        data_collection.update({fields[32]: y_rightEye})
        x_rightEye = landmarks[5].x
        data_collection.update({fields[33]: y_rightEye})
        # 6 - right eye (outer)
        y_rightEye_outer = landmarks[6].y
        data_collection.update({fields[34]: y_rightEye_outer})
        x_rightEye_outer = landmarks[6].x
        data_collection.update({fields[35]: y_rightEye_outer})
        # 7 - left ear
        y_leftEar = landmarks[7].y
        data_collection.update({fields[36]: y_leftEar})
        x_leftEar = landmarks[7].x
        data_collection.update({fields[37]: x_leftEar})
        # 8 - right ear
        y_rightEar = landmarks[8].y
        data_collection.update({fields[38]: y_rightEar})
        x_rightEar = landmarks[8].x
        data_collection.update({fields[39]: x_rightEar})
        # 9 - mouth (left)
        y_leftMouth = landmarks[9].y
        data_collection.update({fields[40]: y_leftMouth})
        x_leftMouth = landmarks[9].x
        data_collection.update({fields[41]: x_leftMouth})
        # 10 - mouth (right)
        y_rightMouth = landmarks[10].y
        data_collection.update({fields[42]: y_rightMouth})
        x_rightMouth = landmarks[10].x
        data_collection.update({fields[43]: x_rightMouth})
        # 13 - left elbow
        data_collection.update({fields[44]: landmarks[13].y})
        data_collection.update({fields[45]: landmarks[13].x})
        # 14 - right elbow
        data_collection.update({fields[46]: landmarks[14].y})
        data_collection.update({fields[47]: landmarks[14].x})
        # 15 - left wrist
        data_collection.update({fields[48]: landmarks[15].y})
        data_collection.update({fields[49]: landmarks[15].x})
        # 16 - right wrist
        data_collection.update({fields[50]: landmarks[16].y})
        data_collection.update({fields[51]: landmarks[16].x})
        # 17 - left pinky
        data_collection.update({fields[52]: landmarks[17].y})
        data_collection.update({fields[53]: landmarks[17].x})
        # 18 - right pinky
        data_collection.update({fields[54]: landmarks[18].y})
        data_collection.update({fields[55]: landmarks[18].x})
        # 19 - left index
        data_collection.update({fields[56]: landmarks[19].y})
        data_collection.update({fields[57]: landmarks[19].x})
        # 20 - right index
        data_collection.update({fields[58]: landmarks[20].y})
        data_collection.update({fields[59]: landmarks[20].x})
        # 21 - left thumb
        data_collection.update({fields[60]: landmarks[21].y})
        data_collection.update({fields[61]: landmarks[21].x})
        # 22 - right thumb
        data_collection.update({fields[62]: landmarks[22].y})
        data_collection.update({fields[63]: landmarks[22].x})


        print(data_collection[fields[0]])
        # prints current time
        values.append(data_collection.copy())

        # Render detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # This is used for the circles/dots
            mp_drawing.DrawingSpec(color=(43, 80, 200), thickness=5, circle_radius=2),
            # This is used for the lines
            mp_drawing.DrawingSpec(color=(96, 25, 93), thickness=5),
        )

        cv2.imshow("Mediapipe Feed", image)

        # Press "x" or "q" to Exit Program/Camera
        if cv2.waitKey(10) & 0xFF == ord("q") or cv2.waitKey(10) & 0xFF == ord("x"):
            df = pd.DataFrame(values)
            df.to_csv(directory + "/" + filename)
            break

cap.release()
cv2.destroyAllWindows()
