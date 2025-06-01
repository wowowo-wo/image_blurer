import cv2
from image_blurer.blurer import rotational_blur
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path of input image")
    parser.add_argument("output", help="name of output file", default="output.jpg")
    parser.add_argument("--steps", type=int, default=20, help="number of rotation steps")
    parser.add_argument("--angle", type=int, default=10, help="max angle of rotation")
    parser.add_argument("--freq", type=int, default=0, help="frequency of noise")
    parser.add_argument("--strength", type=float, default=0, help="strength of noise")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("failed to load image")
        exit(1)

    result = rotational_blur(img, args.steps, args.angle, args.freq, args.strength)
    cv2.imwrite(args.output, result)
    print("finished", args.output)