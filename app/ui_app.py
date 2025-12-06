import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoder import build_encoder
from src.decoder import build_decoder

# Path to decoder weights saved by train.py
MODEL_PATH = "src/models/decoder_checkpoints/decoder_final.h5"
IMG_SIZE = (224, 224)


#GPU
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU detected; running on CPU.")


#Model
@st.cache_resource
def load_models():
    """
    Build encoder & decoder architectures and load decoder weights.

    We saved only weights with `decoder.save_weights(...)` in train.py,
    so we must rebuild the same architecture and then call `load_weights()`.
    """
    configure_gpu()

    # Build encoder and freeze it
    encoder = build_encoder()
    encoder.trainable = False

    # Build decoder matching encoder feature shape
    enc_feature_shape = encoder.output_shape[1:]  # e.g. (56, 56, 256)
    decoder = build_decoder(input_shape=enc_feature_shape)

    # Load decoder weights
    if not os.path.isfile(MODEL_PATH):
        st.error(
            f"Decoder weights not found at: {MODEL_PATH}\n"
            f"Train the model first so decoder_final.h5 is created."
        )
    else:
        decoder.load_weights(MODEL_PATH)
        print(f"Loaded decoder weights from: {MODEL_PATH}")

    return encoder, decoder


#Helper functions
def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a normalized float32 array of shape (224, 224, 3)
    with values in [0, 1].
    """
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMG_SIZE)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return arr


def reconstruct_image(encoder, decoder, image_array: np.ndarray) -> np.ndarray:
    """
    Forward pass: image -> encoder -> decoder.

    image_array: (H, W, 3) float32 in [0, 1]
    returns: reconstructed image (H, W, 3) float32 in [0, 1]
    """
    batch = np.expand_dims(image_array, axis=0)  # (1, H, W, 3)
    features = encoder(batch, training=False)
    recon = decoder(features, training=False).numpy()[0]
    recon = np.clip(recon, 0.0, 1.0)
    return recon


def compute_metrics(orig: np.ndarray, recon: np.ndarray):
    """
    Compute MSE, PSNR, SSIM assuming inputs are float in [0, 1].
    """
    mse_val = float(np.mean((orig - recon) ** 2))
    psnr_val = float(psnr(orig, recon, data_range=1.0))
    ssim_val = float(ssim(orig, recon, data_range=1.0, channel_axis=-1))
    return mse_val, psnr_val, ssim_val


#UI-Image upload

def image_upload_ui(encoder, decoder):
    st.subheader("Image Upload")

    uploaded_file = st.file_uploader(
        "Upload a face image (JPEG/PNG). It will be resized to 224×224.",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to see the reconstruction.")
        return

    pil_img = Image.open(uploaded_file)
    orig_arr = preprocess_pil_image(pil_img)
    recon_arr = reconstruct_image(encoder, decoder, orig_arr)
    mse_val, psnr_val, ssim_val = compute_metrics(orig_arr, recon_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(orig_arr, clamp=True, use_column_width=True)
    with col2:
        st.markdown("**Reconstruction**")
        st.image(recon_arr, clamp=True, use_column_width=True)

    st.markdown(
        f"**MSE:** {mse_val:.6f}  \n"
        f"**PSNR:** {psnr_val:.2f} dB  \n"
        f"**SSIM:** {ssim_val:.3f}"
    )


#UI-webcam

def webcam_ui(encoder, decoder):
    """
    Optional webcam demo using streamlit-webrtc (new API).
    It reconstructs each frame through the encoder+decoder.
    """
    st.subheader("Webcam (Optional)")

    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av
    except ImportError:
        st.error(
            "streamlit-webrtc or av is not installed. "
            "Install them to use webcam mode."
        )
        return

    class ReconstructionProcessor(VideoProcessorBase):
        def __init__(self):
            self.encoder = encoder
            self.decoder = decoder

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # frame: av.VideoFrame -> numpy array (H, W, 3) in BGR
            img_bgr = frame.to_ndarray(format="bgr24")
            img_rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
            pil = Image.fromarray(img_rgb)

            arr = preprocess_pil_image(pil)
            recon = reconstruct_image(self.encoder, self.decoder, arr)
            recon_uint8 = (recon * 255).astype(np.uint8)

            # streamlit-webrtc expects BGR frames
            recon_bgr = recon_uint8[:, :, ::-1]
            return av.VideoFrame.from_ndarray(recon_bgr, format="bgr24")

    st.info("This mode reconstructs each webcam frame using the trained decoder.")

    webrtc_streamer(
        key="reconstruction-webcam",
        video_processor_factory=ReconstructionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )



def main():
    st.set_page_config(
        page_title="Feature-based Image Reconstruction",
        layout="wide",
    )

    st.title("Image Reconstruction from Pretrained Features (Small CNN Decoder)")
    st.write(
        "This demo uses a frozen encoder to extract feature maps and a small "
        "decoder network to reconstruct the input image from those features."
    )

    encoder, decoder = load_models()

    mode = st.sidebar.radio(
        "Choose mode",
        ["Image upload", "Webcam (optional)"],
        index=0,
    )

    if mode == "Image upload":
        image_upload_ui(encoder, decoder)
    else:
        webcam_ui(encoder, decoder)


if __name__ == "__main__":
    main()
