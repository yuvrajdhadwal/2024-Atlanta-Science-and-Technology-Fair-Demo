import cv2
import csv
from datetime import datetime
import numpy as np
import glob
import os

"""
This script uses OpenCV library to create a video file that shows a dot that moves.
Intention is for this script to run alongside data_collection.py. This program is meant
to provide the labels for the features collected in the other file.

Keane Zhang 2/25/2024

'''
modelType = 'Ducking'  # rename based on the folder you want to save to
folder = './' + modelType + './Labels'
csv_filename = folder + './point_coordinates_001.csv'  # Rename this for every new time collecting data
=======
"""


# use glob to find the latest .csv file, then increment the number by 1

files = glob.glob("data/*.csv")
if len(files) == 0:
    csv_filename = "point_coordinates_001.csv"
else:
    files.sort()
    filename = files[-1]
    filename = filename.split(".")[0]
    filename = filename.split("_")
    filename = filename[-1]
    filename = int(filename) + 1
    csv_filename = "point_coordinates_" + str(filename).zfill(3) + ".csv"

directory = "data"

# if the directory does not exist, then the program will create the directory

try:
    os.mkdir(directory)
except FileExistsError:
    pass



# Parameters
width, height = 640, 480
point_radius = 5
color = (0, 0, 255)  # Red in BGR
background_color = (255, 255, 255)  # White in BGR
speed = 8  # Speed of the point's movement

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

# Prepare CSV file
with open(directory + "/" + csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "X", "Y"])  # Header


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
current_position = np.array(route_points[0])
target_index = 1

# Real-time display and writing to CSV
while True:
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

    # Write system time and coordinates to CSV
    with open(directory + "/" + csv_filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%H:%M:%S"), *current_position])

    # Display the frame
    cv2.imshow("Moving Point", canvas)
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord("q"):  # 30 FPS display and 'q' to quit
        break

cv2.destroyAllWindows()
