import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ----------------------------------
# CONFIGURATION
# ----------------------------------
st.set_page_config(page_title="Live Image Reconstruction", layout="wide")
st.title("Live Image Reconstruction from Deep CNN Features")

MODEL_PATH = "D:/opencv-project/models/decoder_checkpoints/decoder_final.h5"

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled")
    except:
        pass

# ----------------------------------
# LOAD MODELS (encoder + decoder)
# ----------------------------------
@st.cache_resource
def load_models():
    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    encoder = tf.keras.Model(inputs=base_model.input,
                             outputs=base_model.get_layer('block3_conv3').output)
    decoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return encoder, decoder

encoder, decoder = load_models()

# ----------------------------------
# SIDEBAR CONTROLS
# ----------------------------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Input mode:", ["Upload Image", "Live Camera"])
scale_factor = st.sidebar.slider("Output Intensity Scale", 0.5, 2.0, 1.0, 0.1)

# ----------------------------------
# PROCESS FUNCTION
# ----------------------------------
def reconstruct_image(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Extract features
    features = encoder.predict(img_batch)
    features_resized = tf.image.resize(features, (32, 32))

    # Decode
    reconstructed = decoder.predict(features_resized)
    reconstructed = tf.image.resize(reconstructed, (224, 224))
    recon_img = np.clip(reconstructed[0].numpy() * scale_factor, 0, 1)
    return recon_img

# ----------------------------------
# UPLOAD IMAGE MODE
# ----------------------------------
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original Image", use_column_width=True)
        with col2:
            with st.spinner("Reconstructing..."):
                reconstructed = reconstruct_image(img_array)
                st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)
        st.success("Done!")

# ----------------------------------
# LIVE CAMERA MODE
# ----------------------------------
elif mode == "Live Camera":

    class LiveReconstructor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            recon = reconstruct_image(img_rgb)
            # Combine original + reconstructed side by side
            recon_bgr = cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            combined = np.hstack((img, recon_bgr))
            return combined

    st.markdown("**Tip:** Allow camera access when prompted. You’ll see live reconstruction in the stream.")
    webrtc_streamer(key="reconstruct", video_transformer_factory=LiveReconstructor)
