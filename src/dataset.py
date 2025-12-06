import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import tensorflow as tf

DATA_DIR = "/Users/goutham/PycharmProjects/ComputerVision/dataset/celeba_hq"


def load_celeba_hq(batch_size=8, img_size=(224, 224)):
    """
    Loads CelebA-HQ (or any image folder) as a tf.data.Dataset.

    Expected structure:

        DATA_DIR/
            any_class_name/
                img1.jpg
                img2.jpg
            another_class/
                img3.jpg
                ...

    Labels are ignored; we only care about images.
    """

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"DATA_DIR does not exist: {DATA_DIR}\n"
            f"Please update DATA_DIR in src/dataset.py."
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels=None,
        label_mode=None,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
    )

    # Normalize to [0,1]
    ds = ds.map(lambda x: tf.cast(x, tf.float32) / 255.0)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    print("Testing dataset loader...")
    ds = load_celeba_hq(batch_size=4, img_size=(224, 224))
    batch = next(iter(ds))
    print("Loaded batch shape:", batch.shape)
    print("Pixel range:", tf.reduce_min(batch).numpy(), "-", tf.reduce_max(batch).numpy())
