import streamlit as st
import numpy as np
import cv2
from PIL import Image
from image_blurer.blurer import (
    rotational_blur,
    zoom_blur,
    motion_blur,
    handshake_blur
)

st.set_page_config(page_title="Image Blurer", layout="centered")

st.title("Image Blurer GUI Tool")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png","webp"])
blur_type = st.selectbox("Select blur type", ["rotational", "zoom", "motion", "handshake"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Original Image", use_container_width=True)

    if blur_type == "rotational":
        st.subheader("Rotational Blur Parameters")
        num_steps = st.slider("Number of steps (samples along rotation arc)", 1, 100, 20)
        angle = st.slider("Rotation angle (degrees)", 1, 45, 10)
        noise_freq = st.number_input("Noise frequency (0 = no noise)", 0, 100, 0)
        noise_str = st.number_input("Noise strength (standard deviation)", 0.0, 100.0, 0.0)

        if st.button("Apply Blur"):
            result = rotational_blur(image_np, num_steps, angle, noise_freq, noise_str)
            st.image(result, caption="Blurred Image", use_container_width=True)

    elif blur_type == "zoom":
        st.subheader("Zoom Blur Parameters")
        num_steps = st.slider("Number of steps (samples along zoom path)", 1, 100, 20)
        zoom_strength = st.slider("Zoom strength (e.g. 1.02 = slight zoom)", 1.00, 3.0, 1.02)
        noise_freq = st.number_input("Noise frequency (0 = no noise)", 0, 100, 0)
        noise_str = st.number_input("Noise strength (standard deviation)", 0.0, 100.0, 0.0)

        if st.button("Apply Blur"):
            result = zoom_blur(image_np, num_steps, zoom_strength, noise_freq, noise_str)
            st.image(result, caption="Blurred Image", use_container_width=True)

    elif blur_type == "motion":
        st.subheader("Motion Blur Parameters")
        size = st.slider("Kernel size (must be odd)", 3, 101, 15, step=2)
        angle = st.slider("Motion direction angle (degrees)", 0, 180, 0)
        pre_noise_str = st.number_input("Pre-blur noise strength", 0.0, 100.0, 0.0)
        post_noise_str = st.number_input("Post-blur noise strength", 0.0, 100.0, 0.0)

        if st.button("Apply Blur"):
            img = image_np.copy()
            if pre_noise_str > 0:
                noise = np.random.normal(0, pre_noise_str, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            result = motion_blur(img, size=size, angle=angle)

            if post_noise_str > 0:
                noise = np.random.normal(0, post_noise_str, result.shape).astype(np.float32)
                result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            st.image(result, caption="Blurred Image", use_container_width=True)

    elif blur_type == "handshake":
        st.subheader("Handshake Blur Parameters")
        num_steps = st.slider("Number of steps (random shifts)", 1, 100, 30)
        max_shift = st.slider("Maximum shift per step (pixels)", 1, 200, 3)
        noise_freq = st.number_input("Noise frequency (0 = no noise)", 0, 100, 0)
        noise_str = st.number_input("Noise strength (standard deviation)", 0.0, 100.0, 0.0)

        if st.button("Apply Blur"):
            result = handshake_blur(image_np, num_steps, max_shift, noise_freq, noise_str)
            st.image(result, caption="Blurred Image", use_container_width=True)
