# this scripts runs both data_collection and video.py to collect data

import os
import subprocess
import time

# run data_collection.py
subprocess.Popen(["python", "data_collection.py"])

# run video.py

time.sleep(1)

subprocess.Popen(["python", "video.py"])

# upon pressing q, the program will stop
# the data will be collected and saved to the excel file
# the video will be saved to the data folder
# the program will terminate

# terminate
print("Program terminated")
