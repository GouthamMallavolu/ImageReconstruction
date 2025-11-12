import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from src.encoder import build_encoder
from src.decoder import build_decoder
from src.dataset import load_celeba_hq


def test_decoder(model_path="models/decoder_checkpoints/decoder_final.h5",
                 num_samples=4,
                 save_dir="results"):
    """Evaluate trained decoder on CelebA-HQ samples."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU detected: memory growth enabled.")
    else:
        print("No GPU detected. Running on CPU.")

    decoder = tf.keras.models.load_model(model_path, compile=False)
    encoder = build_encoder()
    encoder.trainable = False

    test_ds = load_celeba_hq(batch_size=num_samples, shuffle=False)
    test_images, _ = next(iter(test_ds))

    encoded = encoder(test_images)
    # Resize encoder output to match decoder input
    encoded_resized = tf.image.resize(encoded, (32, 32))
    reconstructed = decoder(encoded_resized)
    # Resize decoder output to match original image size
    reconstructed = tf.image.resize(reconstructed, (224, 224))
    reconstructed = tf.clip_by_value(reconstructed, 0.0, 1.0)

    test_images_np = test_images.numpy()
    reconstructed_np = reconstructed.numpy()

    mse_vals, ssim_vals, psnr_vals = [], [], []

    for i in range(num_samples):
        mse_val = np.mean((test_images_np[i] - reconstructed_np[i]) ** 2)
        ssim_val = ssim(test_images_np[i], reconstructed_np[i],
                        channel_axis=-1, data_range=1.0)
        psnr_val = psnr(test_images_np[i], reconstructed_np[i],
                        data_range=1.0)
        mse_vals.append(mse_val)
        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)

    print(f"\nEvaluation on {num_samples} samples:")
    print(f"  Average MSE  : {np.mean(mse_vals):.6f}")
    print(f"  Average SSIM : {np.mean(ssim_vals):.4f}")
    print(f"  Average PSNR : {np.mean(psnr_vals):.2f} dB")

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    for i in range(num_samples):
        axes[0, i].imshow(np.clip(test_images_np[i], 0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(np.clip(reconstructed_np[i], 0, 1))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

        comparison_path = os.path.join(save_dir, f"comparison_{i+1}.png")
        fig_i, ax_i = plt.subplots(1, 2, figsize=(6, 3))
        ax_i[0].imshow(np.clip(test_images_np[i], 0, 1))
        ax_i[0].set_title("Original")
        ax_i[0].axis("off")
        ax_i[1].imshow(np.clip(reconstructed_np[i], 0, 1))
        ax_i[1].set_title("Reconstructed")
        ax_i[1].axis("off")
        plt.tight_layout()
        fig_i.savefig(comparison_path)
        plt.close(fig_i)

    plt.tight_layout()
    plt.show()
    print(f"Saved comparison images to: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    test_decoder(
        model_path="D:/opencv-project/models/decoder_checkpoints/decoder_final.h5",
        num_samples=4,
        save_dir="results"
    )
