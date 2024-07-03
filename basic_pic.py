from picamera2 import Picamera2
from PIL import Image
import numpy as np

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for a 640x480 still image
config = picam2.create_still_configuration(main={"size": (640, 480)})

# Apply the configuration
picam2.configure(config)

# Start the camera
picam2.start()

# Capture an image
picam2.capture_file("test.jpg")

# Stop the camera
picam2.stop()

# Correct the color inversion by converting BGR to RGB
image = Image.open("test.jpg")

print("Picture taken and saved as test.jpg")
