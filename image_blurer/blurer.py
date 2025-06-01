import cv2
import numpy as np
from .utils import padding_image

def rotational_blur(image, num_steps=20, angle=10, noise_freq=0, noise_str=0):
    h, w = image.shape[:2]
    padding = angle * 10
    padded = padding_image(image, padding)

    padded_h, padded_w = padded.shape[:2]
    center = (padded_w // 2, padded_h // 2)

    angles = np.linspace(-angle, angle, num_steps)
    acc = np.zeros_like(padded,dtype=np.float32)

    for i, a in enumerate(angles):
        M = cv2.getRotationMatrix2D(center,a,1.0)
        rotated = cv2.warpAffine(padded,M,(padded_w,padded_h),flags=cv2.INTER_LINEAR)

        if noise_freq > 0 and i % noise_freq == 0:
            noise = np.random.normal(loc=0,scale=noise_str,size=rotated.shape).astype(np.float32)
            rotated = np.clip(rotated.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        acc += rotated.astype(np.float32)
    
    blurred_padded = (acc / num_steps).astype(np.uint8)
    blurred = blurred_padded[padding:padding+h,padding:padding+w]

    return blurred