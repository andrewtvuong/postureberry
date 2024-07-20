import argparse
from picamera2 import Picamera2
from PIL import Image, ImageDraw
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate
from datetime import datetime
import os

_NUM_KEYPOINTS = 17

def capture_image(output_dir):
    # Initialize the camera
    picam2 = Picamera2()
    
    # Configure the camera for a 640x480 still image
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    
    # Apply the configuration
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    
    # Capture an image
    raw_photo_path = os.path.join(output_dir, "raw_photo.jpg")
    picam2.capture_file(raw_photo_path)
    
    # Stop the camera
    picam2.stop()
    
    print(f"Picture taken and saved as {raw_photo_path}")
    return raw_photo_path

def create_output_path(base_path):
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    date_folder = now.strftime("%Y-%m-%d")
    output_dir = os.path.join(base_path, year, month)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_index = len(os.listdir(output_dir))
    output_path = os.path.join(output_dir, f"{date_folder}-movenet_result_{file_index + 1}.jpg")
    return output_path

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', default=os.path.join(script_dir, 'movenet_single_pose_thunder_ptq_edgetpu.tflite'), help='File path of .tflite file.')
    parser.add_argument(
        '--output',
        default=os.path.join(script_dir, 'output'),
        help='Base directory for the output image.')
    args = parser.parse_args()

    # Capture the image
    raw_photo_path = capture_image(args.output)

    # Load the TFLite model and allocate tensors.
    interpreter = Interpreter(
        model_path=args.model,
        experimental_delegates=[load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load image and preprocess it
    img = Image.open(raw_photo_path)
    width, height = img.size
    img_resized = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]), Image.LANCZOS)
    img_np = np.array(img_resized)
    img_np = np.expand_dims(img_np, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_np)

    # Run the inference
    interpreter.invoke()

    # Extract the output results
    pose = interpreter.get_tensor(output_details[0]['index'])[0].reshape(_NUM_KEYPOINTS, 3)

    # Draw keypoints on the original image
    draw = ImageDraw.Draw(img)
    for i in range(_NUM_KEYPOINTS):
        x, y = pose[i][1] * width, pose[i][0] * height
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))

    # Determine the output path and save the output image
    output_path = create_output_path(args.output)
    img.save(output_path)
    print('Done. Results saved at', output_path)

if __name__ == '__main__':
    main()
