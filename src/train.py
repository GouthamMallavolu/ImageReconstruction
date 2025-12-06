import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from src.encoder import build_encoder
from src.decoder import build_decoder
from src.dataset import load_celeba_hq


def build_autoencoder(input_image_shape=(224, 224, 3)):
    """
    Build an autoencoder model:

        inputs (images) -> encoder (frozen) -> decoder -> reconstruction
    """
    # Build encoder and freeze it
    encoder = build_encoder()
    encoder.trainable = False

    # Get encoder output shape, e.g. (56, 56, 256)
    enc_feature_shape = encoder.output_shape[1:]

    # Build decoder that matches encoder feature map shape
    decoder = build_decoder(input_shape=enc_feature_shape)

    # Full autoencoder: image -> encoder -> decoder
    inputs = tf.keras.Input(shape=input_image_shape, name="autoencoder_input")
    features = encoder(inputs)
    recon = decoder(features)

    autoencoder = tf.keras.Model(inputs, recon, name="encoder_decoder")

    return autoencoder, encoder, decoder


def ssim_l1_loss(y_true, y_pred, alpha=0.8):
    """
    Combined MAE + SSIM loss.
    alpha: weight for L1; (1 - alpha) for SSIM component.
    Images are assumed to be in [0, 1].
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # L1 / MAE
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

    # SSIM returns similarity in [-1, 1]; we want a loss in [0, 2]
    ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim_val)  # 0 = perfect, 1 = bad

    return alpha * l1 + (1.0 - alpha) * ssim_loss


def train_model(
    epochs=10,
    batch_size=8,
    save_dir="/Users/goutham/PycharmProjects/ComputerVision/src/models/decoder_checkpoints",
    steps_per_epoch_max=10000,
):
    """
    Train the decoder (via an autoencoder) on CelebA-HQ.

    - epochs:              number of training epochs
    - batch_size:          batch size for the dataset
    - save_dir:            where to store decoder weights and logs
    - steps_per_epoch_max: optional cap on steps/epoch for training time
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Training Decoder Model (Keras model.fit) ===\n")

    # 1. Build autoencoder (encoder frozen, decoder trainable)
    autoencoder, encoder, decoder = build_autoencoder()
    autoencoder.summary()

    # Try to warm-start from existing decoder weights, if they exist
    weight_path = os.path.join(save_dir, "decoder_final.h5")
    if os.path.isfile(weight_path):
        try:
            decoder.load_weights(weight_path)
            print(f"Loaded existing decoder weights from {weight_path} (fine-tuning).")
        except Exception as e:
            print(f"Could not load existing decoder weights: {e}")

    # 2.
    autoencoder.compile(
        optimizer=Adam(1e-4),
        loss=ssim_l1_loss,
    )

    # 3. Load dataset (images only)
    print("Loading dataset...")
    train_ds = load_celeba_hq(batch_size=batch_size, img_size=(224, 224))

    # Turn it into (inputs, targets) = (x, x) for autoencoder training
    train_ds = train_ds.map(lambda x: (x, x))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Determine steps per epoch by dataset cardinality
    cardinality = tf.data.experimental.cardinality(train_ds).numpy()
    if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
        steps_per_epoch = steps_per_epoch_max
    else:
        steps_per_epoch = int(cardinality)

    if steps_per_epoch_max is not None:
        steps_per_epoch = min(steps_per_epoch, steps_per_epoch_max)

    print(f"Steps per epoch: {steps_per_epoch}")

    # 4. Train with standard Keras progress bar
    history = autoencoder.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )

    # 5. Save decoder weights (encoder is fixed)
    final_decoder_path = os.path.join(save_dir, "decoder_final.h5")
    decoder.save_weights(final_decoder_path)
    print(f"\nTraining complete. Decoder weights saved at: {final_decoder_path}")

    # 6. Save loss history
    loss_history = history.history.get("loss", [])
    history_path = os.path.join(save_dir, "loss_history.json")
    with open(history_path, "w") as f:
        json.dump({"loss": loss_history}, f, indent=4)
    print(f"Saved training loss history at: {history_path}")

    # 7. Plot loss curve
    if loss_history:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, marker="o")
        plt.title("Training Loss Curve (MAE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plot_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss curve at: {plot_path}")

    return decoder, history.history


if __name__ == "__main__":
    train_model(epochs=30, batch_size=8)
