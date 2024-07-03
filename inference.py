import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter, load_delegate

_NUM_KEYPOINTS = 17

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', default='movenet_single_pose_thunder_ptq_edgetpu.tflite', help='File path of .tflite file.')
    parser.add_argument(
        '-i', '--input', default='test.jpg', help='Image to be classified.')
    parser.add_argument(
        '--output',
        default='movenet_result.jpg',
        help='File path of the output image.')
    args = parser.parse_args()

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
    img = Image.open(args.input)
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
