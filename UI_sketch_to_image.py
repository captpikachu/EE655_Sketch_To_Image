import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
generator = load_model("generator_sketch2image.h5", compile=False)

def create_sketch(image_np):
    if image_np.shape[-1] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np[..., 0]
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    edges = cv2.Canny(gray, 100, 200)
    sketch = edges.astype(np.float32) / 255.0
    sketch = np.expand_dims(sketch, axis=-1)
    sketch = tf.image.resize(sketch, [128, 128])
    return sketch

st.title("Sketch-to-Image Generator (Your Model)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (128, 128))
    sketch = create_sketch(image_np)
    sketch = tf.expand_dims(sketch, axis=0)
    sketch = (sketch * 2) - 1
    generated = generator(sketch, training=False)[0]
    generated = (generated + 1) / 2
    generated_np = (generated.numpy() * 255).astype(np.uint8)
    if generated_np.shape[-1] == 1:
        generated_np = np.repeat(generated_np, 3, axis=-1)
    gen_image = Image.fromarray(generated_np)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original")
    with col2:
        st.image(sketch[0, ..., 0].numpy(), caption="Sketch", clamp=True)

    with col3:
        st.image(gen_image, caption="Generated")
        
    buf = io.BytesIO()
    gen_image.save(buf, format="PNG")
    st.download_button("Download Output", data=buf.getvalue(), file_name="generated_image.png", mime="image/png")
