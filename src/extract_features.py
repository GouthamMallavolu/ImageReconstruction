import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import tensorflow as tf
from src.encoder import build_encoder
from src.dataset import load_celeba_hq


def extract_and_save_features(save_dir="/Users/goutham/PycharmProjects/ComputerVision/src/data/features", batch_size=8, max_batches=None):
    """
    Extract encoder features for ALL images in CelebA-HQ and save as .npy files.
    Each .npy pair corresponds to one batch (feat_XXXXX.npy, img_XXXXX.npy).

    Args:
        save_dir (str): Directory to store extracted features.
        batch_size (int): Number of images per batch.
        max_batches (int or None): Optional cap for number of batches.
                                   If None, processes the entire dataset.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Initialize encoder ---
    encoder = build_encoder()
    encoder.trainable = False

    # --- Load dataset ---
    dataset = load_celeba_hq(batch_size=batch_size, shuffle=True)
    total_images = 0
    batch_count = 0

    print("Extracting features for full dataset...")

    for i, (images, _) in enumerate(dataset):
        feats = encoder(images).numpy().astype(np.float32)

        # Normalize features to stabilize decoder training
        feats = feats / (np.max(np.abs(feats), axis=(1, 2, 3), keepdims=True) + 1e-8)

        np.save(os.path.join(save_dir, f"feat_{i:05d}.npy"), feats)
        np.save(os.path.join(save_dir, f"img_{i:05d}.npy"), images.numpy())

        batch_count += 1
        total_images += images.shape[0]

        if (i + 1) % 20 == 0:
            print(f"Saved {i + 1} batches ({total_images} images)")

        # If max_batches is specified, stop early
        if max_batches is not None and (i + 1) >= max_batches:
            break

    print(f"\nFeature extraction complete.")
    print(f"Total batches saved: {batch_count}")
    print(f"Total images processed: {total_images}")


if __name__ == "__main__":
    extract_and_save_features(save_dir="/Users/goutham/PycharmProjects/ComputerVision/src/data/features", batch_size=8, max_batches=None)
