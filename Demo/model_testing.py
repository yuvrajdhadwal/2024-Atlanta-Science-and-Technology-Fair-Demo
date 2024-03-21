import xgboost as xgb
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp

"""
This script is simply used to test the accuracy of the model we have created in
model_training.py. This script will not be used for the exhibit. Nor does it have any pupose
except to let us know that our model is working well.

This file is not complete

Yuvraj Dhadwal 2/25/24
"""

model = xgb.XGBRegressor({"nthread": 4})
model.load_model("pose_recognition_model.bin")

# Parameters
width, height = 640, 480
point_radius = 15
color = (0, 0, 255)  # Red in BGR
model_color = (255, 0, 0)
background_color = (255, 255, 255)  # White in BGR
speed = 8  # Speed of the point's movement
model_speed = 30

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# Define the route points based on the image provided
route_points = [
    (320, 20),  # Starting point (approximate center top)
    (10, 20),
    (620, 20),
    (320, 20),
    (10, 60),
    (320, 20),
    (620, 60),
    (320, 20),
    (10, 120),
    (320, 20),
    (620, 120),
    (320, 20),
    (10, 180),
    (320, 20),
    (620, 180),
    (320, 20),
    (10, 240),
    (320, 20),
    (620, 240),
    (320, 20),
    (10, 300),
    (320, 20),
    (620, 300),
    (320, 20),
    (10, 360),
    (320, 20),
    (620, 360),
    (320, 20),
    (10, 420),
    (320, 20),
    (620, 420),
    (320, 20),
    (10, 460),
    (320, 20),
    (620, 460),
    (320, 20),
    (160, 460),
    (320, 20),
    (480, 460),
    (320, 20),
    (320, 460),  # Ending point (approximate center bottom)
]

# Function to move the point from current to target position
def move_point(current, target, speed):
    current = np.array(current, dtype=np.float32)
    target = np.array(target, dtype=np.float32)

    direction = target - current
    distance = np.linalg.norm(direction)

    # Normalize the direction vector
    if distance != 0:
        direction = direction / distance

    # Move by the speed along the direction, but do not overshoot the target
    step = direction * min(speed, distance)

    # Update the current position
    new_position = current + step

    # Round to the nearest integer to ensure that the point moves in both x and y directions
    new_position = np.round(new_position).astype(int)

    return new_position


# Main loop
current_position, current_prediction = np.array(route_points[0])
target_index = 1
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Real-time display and writing to CSV
    while cap.isOpened():
        data_collection = {}
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
        data_collection.update({fields[0]: 0})
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

        values = []
        values.append(data_collection.copy())

        data = pd.DataFrame(values)

        data["Hip Width"] = abs(data["x_rightHip"] - data["x_leftHip"])
        data["Torso Height"] = abs(data["y_leftEye"] - data["y_leftHip"])
        data["Shoulder Width"] = abs(
            data["x_leftShoulder"] - data["x_rightShoulder"]
        )
        data["Leg Length"] = abs(data["y_leftFoot"] - data["y_leftHip"])
        data["Height"] = abs(data["y_leftEye"] - data["y_leftFoot"])
        data["Height to Hips Ratio"] = abs(data["Height"] / data["Hip Width"])
        data = data.iloc[:, 1:]

        prediction = model.predict(data)
        prediction_points = pd.DataFrame(prediction, columns=["X", "Y"])
        pred_x = prediction_points.iloc[0]['X']
        pred_y = prediction_points.iloc[0]['Y']
        pred_point = pred_x, pred_y
        print(pred_point)
        current_prediction = move_point(current_prediction, pred_point, model_speed)

        canvas = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Move the point
        current_position = move_point(current_position, route_points[target_index], speed)

        # Check if the point has reached the next target
        if np.array_equal(current_position, route_points[target_index]):
            target_index += 1
            if target_index == len(route_points):
                break  # End if last point is reached

        # Draw the point
        cv2.circle(canvas, tuple(current_position), point_radius, color, -1)
        cv2.circle(canvas, tuple(current_prediction), point_radius, model_color, -1)
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

        # Display the frame
        cv2.imshow("Moving Point", canvas)
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord("q"):  # 30 FPS display and 'q' to quit
            break

cv2.destroyAllWindows()
