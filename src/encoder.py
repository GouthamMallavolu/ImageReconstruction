import tensorflow as tf
from tensorflow.keras.applications import VGG16


def build_encoder():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze encoder layers
    for layer in base_model.layers:
        layer.trainable = False
    # Use intermediate feature layer
    encoder = tf.keras.Model(inputs=base_model.input,
                             outputs=base_model.get_layer("block3_conv3").output,
                             name="encoder")
    return encoder
