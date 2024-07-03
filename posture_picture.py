import argparse
from picamera2 import Picamera2
from PIL import Image, ImageDraw
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

_NUM_KEYPOINTS = 17

def capture_image():
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
    
    print("Picture taken and saved as test.jpg")

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', default='movenet_single_pose_thunder_ptq_edgetpu.tflite', help='File path of .tflite file.')
    parser.add_argument(
        '--output',
        default='movenet_result.jpg',
        help='File path of the output image.')
    args = parser.parse_args()

    # Capture the image
    capture_image()

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
    img = Image.open("test.jpg")
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

    # Save the output image
    img.save(args.output)
    print('Done. Results saved at', args.output)

if __name__ == '__main__':
    main()
