import os
import numpy as np
import tensorflow as tf
from src.decoder import build_decoder
from src.extract_features import extract_and_save_features


def make_dataset(feature_dir):
    """
    Build a tf.data.Dataset that streams all precomputed feature and image .npy batches.
    """
    feature_files = sorted([f for f in os.listdir(feature_dir) if f.startswith("feat_")])
    image_files = sorted([f for f in os.listdir(feature_dir) if f.startswith("img_")])

    if len(feature_files) == 0 or len(image_files) == 0:
        raise FileNotFoundError("No precomputed .npy feature or image files found in data/features")

    # Load one batch to infer shapes
    sample_feat = np.load(os.path.join(feature_dir, feature_files[0]))
    sample_img = np.load(os.path.join(feature_dir, image_files[0]))

    def generator():
        """Generator to load features and images batch-by-batch from disk."""
        for f_feat, f_img in zip(feature_files, image_files):
            X_feats = np.load(os.path.join(feature_dir, f_feat)).astype(np.float32)
            Y_imgs = np.load(os.path.join(feature_dir, f_img)).astype(np.float32)
            yield X_feats, Y_imgs

    # Stream batches from disk and prefetch to keep GPU fed
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *sample_feat.shape[1:]), dtype=tf.float32),
            tf.TensorSpec(shape=(None, *sample_img.shape[1:]), dtype=tf.float32),
        ),
    ).repeat().prefetch(tf.data.AUTOTUNE)

    return ds, sample_feat.shape[1:], len(feature_files)


def train_model(epochs=20, batch_size=8, feature_dir="D:/opencv-project/src/data/features",
                save_dir="models/decoder_checkpoints"):
    """
    Train the decoder network using all precomputed features from the dataset.
    Automatically adjusts to number of batches present in feature_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- GPU configuration ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU with float32 precision.")
    else:
        print("No GPU found — running on CPU.")

    # --- If no features exist, extract them from full dataset ---
    if not os.path.exists(feature_dir) or len(os.listdir(feature_dir)) == 0:
        print("No precomputed features found. Extracting from dataset...")
        extract_and_save_features(save_dir=feature_dir, batch_size=batch_size, max_batches=None)

    # --- Prepare dataset ---
    ds, input_shape, num_batches = make_dataset(feature_dir)

    print(f"Found {num_batches} feature-image batch pairs in '{feature_dir}'.")

    # --- Build and compile decoder ---
    decoder = build_decoder(input_shape=input_shape)
    decoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    print(f"Starting training for {epochs} epochs ({num_batches} steps/epoch)...")

    # --- Train model ---
    history = decoder.fit(
        ds,
        epochs=epochs,
        steps_per_epoch=num_batches,
        verbose=1  # Show Keras progress bar
    )

    # --- Save trained model ---
    model_path = os.path.join(save_dir, "decoder_final.h5")
    decoder.save(model_path)
    print("Training complete.")
    print("Model saved at:", os.path.abspath(model_path))

    return decoder, history


if __name__ == "__main__":
    train_model(epochs=20, batch_size=8)
