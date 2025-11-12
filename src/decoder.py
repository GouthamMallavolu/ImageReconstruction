import tensorflow as tf
from tensorflow.keras import layers, models


def build_decoder(input_shape=(56, 56, 256)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(inputs)  # 56 → 112
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)       # 112 → 224
    x = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)                         # final 224×224×3
    model = models.Model(inputs, x, name="decoder")
    return model
