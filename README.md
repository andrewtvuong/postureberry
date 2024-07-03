# Posture Detection with Coral TPU and Raspberry Pi

This project utilizes a Coral TPU USB Accelerator to detect and alert you about your posture using a Raspberry Pi and a camera. 

The effects of bad posture and modern desk jobs can be very damaging to our health. This project aims to seamlessly and inexpensively help you maintain good posture by alerting you when you are slouching or not sitting correctly. Can extend beyond alerts for bad posture detected => ???? alert, punishment, 📉 etc ????

## Prerequisites

### TPU USB Accelerator Compatibility (OPTIONAL, CPU Models available too)

The Coral TPU USB Accelerator's official support for `tflite-runtime` is limited to version 11.0, while the current available version is 14.0 on pip. To resolve this, you need to install a custom `libedgetpu` package.

### Custom libedgetpu Installation

Use the following custom `libedgetpu` package, as the official support has been abandoned by Google:

- Custom `libedgetpu` package: [libedgetpu v16.0TF2.13.1-2](https://github.com/feranick/libedgetpu/releases/tag/v16.0TF2.13.1-2)
- Relevant discussion: [Google Coral pycoral issue #137](https://github.com/google-coral/pycoral/issues/137)

Ensure you are on the latest version of Rasbian (Bookworm) for the best compatibility.

## Installation

1. Install the custom `libedgetpu` package from the provided link.
2. Install `tflite-runtime` version 2.14 as it is current as of July 2, 2024.

## Usage

### posture_picture.py

The `posture_picture.py` script captures an image and runs the posture detection model on it.

### stream.py

The `stream.py` script works well for previewing video to tune the camera positioning.

### Notes

- **PIL (Python Imaging Library)**: Using PIL is vastly superior to OpenCV for image processing tasks in this project. OpenCV did not produce the desired results for posture detection annotations, scaling was off.
- **Model**: The model used https://coral.ai/models/pose-estimation/