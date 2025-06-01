import argparse
import os
import cv2
import numpy as np

from .blurer import (
    rotational_blur,
    zoom_blur,
    motion_blur,
    handshake_blur
)

def parse_args():
    parser = argparse.ArgumentParser(description="Apply various types of blur to an image.")

    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path", default="output.jpg")

    parser.add_argument("--type", required=True, choices=["rotational", "zoom", "motion", "handshake"], help="Type of blur to apply")

    parser.add_argument("--noise_freq", type=int, default=0, help="Noise frequency (every N steps)")
    parser.add_argument("--noise_str", type=float, default=0.0, help="Noise strength")

    parser.add_argument("--angle", type=float, default=10.0, help="Rotation angle for rotational blur")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of steps for blur accumulation")

    parser.add_argument("--zoom_strength", type=float, default=1.02, help="Zoom strength for zoom blur")

    parser.add_argument("--size", type=int, default=15, help="Kernel size for motion blur")
    parser.add_argument("--motion_angle", type=float, default=0.0, help="Angle for motion blur")
    parser.add_argument("--pre_noise_str", type=float, default=0.0, help="Pre-blur noise strength for motion blur")
    parser.add_argument("--post_noise_str", type=float, default=0.0, help="Post-blur noise strength for motion blur")

    parser.add_argument("--max_shift", type=int, default=3, help="Max pixel shift for handshake blur")

    return parser.parse_args()


def run(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Failed to read image: {args.input}")

    if args.type == "rotational":
        result = rotational_blur(
            image,
            num_steps=args.num_steps,
            angle=args.angle,
            noise_freq=args.noise_freq,
            noise_str=args.noise_str
        )

    elif args.type == "zoom":
        result = zoom_blur(
            image,
            num_steps=args.num_steps,
            zoom_strength=args.zoom_strength,
            noise_freq=args.noise_freq,
            noise_str=args.noise_str
        )

    elif args.type == "motion":
        if args.pre_noise_str > 0:
            noise = (np.random.normal(0, args.pre_noise_str, image.shape)).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        result = motion_blur(image, size=args.size, angle=args.motion_angle)

        if args.post_noise_str > 0:
            noise = (np.random.normal(0, args.post_noise_str, result.shape)).astype(np.float32)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif args.type == "handshake":
        result = handshake_blur(
            image,
            num_steps=args.num_steps,
            max_shift=args.max_shift,
            noise_freq=args.noise_freq,
            noise_str=args.noise_str
        )

    else:
        raise ValueError(f"Unsupported blur type: {args.type}")

    cv2.imwrite(args.output, result)
    print(f"Saved blurred image to {args.output}")

