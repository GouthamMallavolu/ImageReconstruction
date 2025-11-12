import tensorflow as tf
import os


def load_celeba_hq(batch_size=8, shuffle=True):
    """
    Load the CelebA-HQ dataset and normalize images to [0, 1].
    Directory structure:
        D:/Datasets/CelebA-HQ/
            ├── class1/
            └── class2/
    (Classes are ignored for this task.)
    """
    data_dir = "D:/opencv-project/dataset/celeba_hq"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    IMG_SIZE = (224, 224)
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle
    ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    return ds


if __name__ == "__main__":
    print("Testing dataset loader...")
    ds = load_celeba_hq(batch_size=4)
    images, _ = next(iter(ds))
    print("Loaded batch shape:", images.shape)
    print("Pixel range:", tf.reduce_min(images).numpy(), "-", tf.reduce_max(images).numpy())
