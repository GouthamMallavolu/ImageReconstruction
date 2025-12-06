import os
import sys
import json
import math
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoder import build_encoder
from src.decoder import build_decoder

try:
    # If dataset.py is under src
    from src.dataset import load_celeba_hq
except ImportError:
    # If dataset.py is at project root
    from dataset import load_celeba_hq


DEFAULT_WEIGHTS = "src/models/decoder_checkpoints/decoder_final.h5"
DEFAULT_SAVE_DIR = "outputs/evaluation"
IMG_SIZE = (224, 224)

#GPU
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled.")
        except Exception as e:  # noqa
            print("Could not set memory growth:", e)
    else:
        print("No GPU detected; running on CPU.")


#Metrics
def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute MSE, PSNR, SSIM for a single pair of images.
    Inputs assumed to be float32 in [0, 1], shape (H, W, 3).
    """
    mse_val = float(np.mean((orig - recon) ** 2))
    psnr_val = float(psnr(orig, recon, data_range=1.0))
    ssim_val = float(ssim(orig, recon, data_range=1.0, channel_axis=-1))
    return mse_val, psnr_val, ssim_val


#Plotting
def plot_histogram(values: List[float], title: str, xlabel: str, save_path: str):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram: {save_path}")


def plot_reconstruction_grid(
    originals: List[np.ndarray],
    reconstructions: List[np.ndarray],
    mse_list: List[float],
    psnr_list: List[float],
    ssim_list: List[float],
    save_path: str,
):
    """
    Show a small grid of original vs reconstruction for qualitative inspection.
    """
    n = len(originals)
    if n == 0:
        return

    cols = n
    rows = 2
    plt.figure(figsize=(4 * cols, 6))

    for i in range(n):
        # Original
        ax1 = plt.subplot(rows, cols, i + 1)
        ax1.imshow(originals[i])
        ax1.axis("off")
        ax1.set_title("Original")

        # Reconstruction
        ax2 = plt.subplot(rows, cols, cols + i + 1)
        ax2.imshow(reconstructions[i])
        ax2.axis("off")
        ax2.set_title(
            f"Recon\nMSE={mse_list[i]:.4f}\nPSNR={psnr_list[i]:.2f} dB\nSSIM={ssim_list[i]:.3f}"
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction grid: {save_path}")


def plot_loss_curve(weights_path: str, save_dir: str):
    """
    If a training loss history JSON exists next to the weights, plot it.
    """
    history_path = os.path.join(os.path.dirname(weights_path), "loss_history.json")
    if not os.path.isfile(history_path):
        print("No loss_history.json found; skipping loss curve plot.")
        return

    with open(history_path, "r") as f:
        hist = json.load(f)

    loss = hist.get("loss")
    if not loss:
        print("loss_history.json has no 'loss' key; skipping.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(loss, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, "training_loss_curve.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training loss curve: {save_path}")



def evaluate_decoder(
    weights_path: str = DEFAULT_WEIGHTS,
    num_samples: int = 200,
    batch_size: int = 8,
    save_dir: str = DEFAULT_SAVE_DIR,
):
    """
    Evaluate the trained decoder on CelebA-HQ.

    Args:
        weights_path: path to decoder weights (.h5) saved by train.py
        num_samples:  number of images to evaluate on
        batch_size:   batch size for the tf.data pipeline
        save_dir:     directory to save metrics and plots
    """
    configure_gpu()

    print("\n=== Building encoder and decoder ===")
    encoder = build_encoder()
    encoder.trainable = False
    feature_shape = encoder.output_shape[1:]
    print(f"Encoder feature shape: {feature_shape}")

    decoder = build_decoder(input_shape=feature_shape)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Decoder weights not found at: {weights_path}\n"
            f"Train the model first so decoder_final.h5 is created."
        )

    decoder.load_weights(weights_path)
    print(f"Loaded decoder weights from: {weights_path}")

    # Dataset
    print("\n=== Loading dataset ===")
    ds = load_celeba_hq(batch_size=batch_size)  # images in [0, 1], shape (224, 224, 3)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    n_batches = math.ceil(num_samples / batch_size)
    print(f"Evaluating on {num_samples} samples (~{n_batches} batches).")

    # Metrics storage
    mse_list: List[float] = []
    psnr_list: List[float] = []
    ssim_list: List[float] = []

    # For qualitative grid
    orig_samples: List[np.ndarray] = []
    recon_samples: List[np.ndarray] = []
    mse_samples: List[float] = []
    psnr_samples: List[float] = []
    ssim_samples: List[float] = []

    processed = 0

    for batch_idx, batch in enumerate(ds.take(n_batches)):
        # If dataset yields (images, labels), unpack; otherwise it's just images
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        images_np = images.numpy().astype(np.float32)
        feats = encoder(images_np, training=False)
        recons_np = decoder(feats, training=False).numpy().astype(np.float32)
        recons_np = np.clip(recons_np, 0.0, 1.0)

        bsz = images_np.shape[0]
        for i in range(bsz):
            if processed >= num_samples:
                break

            orig = images_np[i]
            recon = recons_np[i]

            m_val, p_val, s_val = compute_metrics(orig, recon)
            mse_list.append(m_val)
            psnr_list.append(p_val)
            ssim_list.append(s_val)

            # Store a few examples for visualization
            if len(orig_samples) < 4:
                orig_samples.append(orig)
                recon_samples.append(recon)
                mse_samples.append(m_val)
                psnr_samples.append(p_val)
                ssim_samples.append(s_val)

            processed += 1

        if processed >= num_samples:
            break

    print(f"Processed {processed} images.")


    mse_mean = float(np.mean(mse_list)) if mse_list else float("nan")
    mse_std = float(np.std(mse_list)) if mse_list else float("nan")
    psnr_mean = float(np.mean(psnr_list)) if psnr_list else float("nan")
    psnr_std = float(np.std(psnr_list)) if psnr_list else float("nan")
    ssim_mean = float(np.mean(ssim_list)) if ssim_list else float("nan")
    ssim_std = float(np.std(ssim_list)) if ssim_list else float("nan")

    summary = {
        "num_samples": processed,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                **summary,
                "mse_all": mse_list,
                "psnr_all": psnr_list,
                "ssim_all": ssim_list,
            },
            f,
            indent=4,
        )
    print(f"Saved metrics summary to: {summary_path}")

    if mse_list:
        plot_histogram(
            psnr_list,
            title="PSNR Distribution",
            xlabel="PSNR (dB)",
            save_path=os.path.join(save_dir, "psnr_histogram.png"),
        )
        plot_histogram(
            ssim_list,
            title="SSIM Distribution",
            xlabel="SSIM",
            save_path=os.path.join(save_dir, "ssim_histogram.png"),
        )

        plot_reconstruction_grid(
            orig_samples,
            recon_samples,
            mse_samples,
            psnr_samples,
            ssim_samples,
            save_path=os.path.join(save_dir, "sample_reconstructions.png"),
        )

        plot_loss_curve(weights_path, save_dir)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate decoder on CelebA-HQ (MSE/PSNR/SSIM + plots)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Path to decoder weights (.h5). Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of images to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save metrics and plots. Default: {DEFAULT_SAVE_DIR}",
    )

    args = parser.parse_args()

    evaluate_decoder(
        weights_path=args.weights,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
    )
