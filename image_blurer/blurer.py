import cv2
import numpy as np
from .utils import padding_image


def rotational_blur(image, num_steps=20, angle=10, noise_freq=0, noise_str=0):

    h, w = image.shape[:2]
    padding = int(angle * 10)
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

def zoom_blur(image, num_steps=20, zoom_strength=1.02, noise_freq=0, noise_str=0):

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    padding = 50

    padded = padding_image(image, padding)

    padded_h, padded_w = padded.shape[:2]
    center = (padded_w // 2, padded_h // 2)

    acc = np.zeros_like(padded, dtype=np.float32)

    for i in range(num_steps):
        scale = 1 + (zoom_strength - 1) * (i / num_steps)
        M = cv2.getRotationMatrix2D(center, 0, scale)
        zoomed = cv2.warpAffine(padded, M, (padded_w, padded_h), flags=cv2.INTER_LINEAR)

        if noise_freq > 0 and i % noise_freq == 0:
            noise = np.random.normal(loc=0,scale=noise_str,size=zoomed.shape).astype(np.float32)
            zoomed = np.clip(zoomed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        acc += zoomed.astype(np.float32)

    zoomed_padded = (acc / num_steps).astype(np.uint8)
    zoomed = zoomed_padded[padding:padding+h, padding:padding+w]

    return zoomed

def motion_blur(image, size=15, angle=0, pre_noise_str = 0, post_noise_str = 0):

    kernel = np.zeros((size, size))
    kernel[size // 2, :] = np.ones(size)

    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (size, size))
    kernel /= np.sum(kernel)

    if pre_noise_str > 0:
        noise = np.random.normal(loc=0, scale=pre_noise_str, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    motioned = cv2.filter2D(image, -1 , kernel)

    if post_noise_str > 0:
        noise = np.random.normal(loc=0, scale=post_noise_str, size=motioned.shape).astype(np.float32)
        motioned = np.clip(motioned.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return motioned

def handshake_blur(image, num_steps=30, max_shift=3, noise_freq=0, noise_str=0):

    h, w = image.shape[:2]
    acc = np.zeros_like(image, dtype=np.float32)

    for i in range(num_steps):
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        if noise_freq > 0 and i % noise_freq == 0:
            noise = np.random.normal(loc=0,scale=noise_str,size=shifted.shape).astype(np.float32)
            shifted = np.clip(shifted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        acc += shifted.astype(np.float32)

    handshaked = (acc / num_steps).astype(np.uint8)

    return handshaked