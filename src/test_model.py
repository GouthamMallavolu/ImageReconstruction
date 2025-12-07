import os
import sys

# Make project root importable when running this file directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from src.encoder import build_encoder
from src.decoder import build_decoder
from src.dataset import load_celeba_hq


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU detected: memory growth enabled.")
        except Exception as e:
            print("Could not set memory growth:", e)
    else:
        print("No GPU detected; running on CPU.")


def test_decoder(
    model_dir="src/models/decoder_checkpoints",
    num_samples=4,
    save_dir="src/results/test_run",
):
    """
    Load the trained encoder + decoder, run reconstruction on a few images,
    compute PSNR/SSIM, and save side-by-side comparison plots.
    """
    configure_gpu()

    os.makedirs(save_dir, exist_ok=True)

    # ----- Build encoder/decoder -----
    print("Building encoder...")
    encoder = build_encoder()
    encoder.trainable = False

    enc_feature_shape = encoder.output_shape[1:]
    print("Encoder feature shape:", enc_feature_shape)

    print("Building decoder...")
    decoder = build_decoder(input_shape=enc_feature_shape)

    # ----- Load decoder weights -----
    weight_path = os.path.join(model_dir, "decoder_final.h5")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Decoder weights not found at: {weight_path}\n"
            f"Make sure you have trained the model and saved decoder_final.h5."
        )

    decoder.load_weights(weight_path)
    print(f"Loaded decoder weights from: {weight_path}")

    # ----- Load a small batch of test images -----
    print("Loading test images...")
    test_ds = load_celeba_hq(batch_size=num_samples, img_size=(224, 224))

    # Take one batch
    batch = next(iter(test_ds))

    # IMPORTANT: handle (images,) or (images, labels) tuples
    if isinstance(batch, tuple):
        images = batch[0]
    else:
        images = batch

    test_images = images.numpy()  # shape (B, 224, 224, 3), range [0,1]
    print(f"Test batch shape: {test_images.shape}")

    # ----- Forward pass: encoder -> decoder (NO resizing of features) -----
    print("Running encoder + decoder...")
    encoded = encoder(test_images, training=False)          # (B, Hf, Wf, Cf) e.g. (B,56,56,256)
    reconstructed = decoder(encoded, training=False)        # (B, 224, 224, 3)

    reconstructed_np = reconstructed.numpy()
    print(f"Reconstructed shape: {reconstructed_np.shape}")

    # ----- Compute metrics -----
    mse_list = []
    psnr_list = []
    ssim_list = []

    for i in range(num_samples):
        orig = np.clip(test_images[i], 0.0, 1.0)
        recon = np.clip(reconstructed_np[i], 0.0, 1.0)

        mse_val = np.mean((orig - recon) ** 2)
        psnr_val = psnr(orig, recon, data_range=1.0)
        ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=-1)

        mse_list.append(mse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    print("\n=== Reconstruction Metrics (per-batch average) ===")
    print(f"MSE  : {np.mean(mse_list):.6f}")
    print(f"PSNR : {np.mean(psnr_list):.3f} dB")
    print(f"SSIM : {np.mean(ssim_list):.4f}")

    # ----- Save side-by-side comparison plots -----
    for i in range(num_samples):
        orig = np.clip(test_images[i], 0.0, 1.0)
        recon = np.clip(reconstructed_np[i], 0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(recon)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        fig.suptitle(
            f"Sample {i} | MSE: {mse_list[i]:.5f}, "
            f"PSNR: {psnr_list[i]:.2f}dB, SSIM: {ssim_list[i]:.3f}"
        )

        out_path = os.path.join(save_dir, f"comparison_{i}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved comparison for comparison_{i} to: {out_path}")

    print("\nTest run complete.")


if __name__ == "__main__":
    test_decoder()
