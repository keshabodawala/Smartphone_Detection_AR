# Smarthpnone Detection App

A real-time smartphone detection application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).

## Getting Started
`python object_detection_app_3D.py`
    Optional arguments (default value):
    * Device index of the camera `--source=0`
    * Width of the frames in the video stream `--width=480`
    * Height of the frames in the video stream `--height=360`
    * Path of folder containing animation frames `--animation-path="animation_full"`


## Requirements
- Python 3.5
- TensorFlow 1.2
- OpenCV 3.0
- PyOpenGL 3.1.2
- [Optional: only for 2D app] CUDA 8
