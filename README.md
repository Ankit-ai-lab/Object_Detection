# Object_Detection

YOLOv8 Object Detection on Video

This repository contains a Python script that utilizes the YOLOv8 model for object detection in a video. The detected objects are highlighted with bounding boxes, and the processed video is saved as an MP4 file.

Features

Uses Ultralytics YOLOv8 for object detection.

Processes an input video and detects objects frame by frame.

Saves the output video with bounding boxes and vehicle count.

Supports GPU acceleration for faster processing.

Requirements

Python 3.x

OpenCV

PyTorch

Ultralytics YOLO

Installation

Clone the repository and install the dependencies:

pip install ultralytics opencv-python torch numpy

Usage

Run the script with:

python object_detection.py

Ensure that your input video file is available in the specified path.

Output

The processed video will be saved as output.mp4.
