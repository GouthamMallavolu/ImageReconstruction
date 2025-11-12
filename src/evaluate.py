import matplotlib.pyplot as plt
import tensorflow as tf
from src.encoder import build_encoder
from src.decoder import build_decoder


def visualize_reconstruction(decoder_path, sample_image):
    encoder = build_encoder()
    decoder = tf.keras.models.load_model(decoder_path, compile=False)

    feature = encoder(tf.expand_dims(sample_image, 0))
    reconstruction = decoder(feature)[0]

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(sample_image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Reconstructed")
    plt.imshow(reconstruction)
    plt.axis('off')
    plt.show()
