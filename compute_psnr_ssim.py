import os
import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Directories for generated and original images
generated_dir = "output/awgn/reconstructed_20dB"
original_dir = "output/awgn/original"

# Initialize lists to store metrics
psnr_values = []
ssim_values = []

# Loop through the generated images
for filename in os.listdir(generated_dir):
    # Load the generated and original images
    generated_path = os.path.join(generated_dir, filename)
    original_path = os.path.join(original_dir, filename)

    if not os.path.exists(original_path):
        print(f"Original image not found for: {filename}")
        continue

    # Read images
    generated_image = imread(generated_path)
    original_image = imread(original_path)
    print(f"Original image shape: {original_image.shape}")
    print(f"Generated image shape: {generated_image.shape}")

    # Ensure images are the same size
    if generated_image.shape != original_image.shape:
        print(f"Image shapes do not match for: {filename}")
        continue

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(original_image, generated_image)
    psnr_values.append(psnr)

    # Calculate SSIM
    ssim = structural_similarity(original_image, generated_image, channel_axis=-1)
    ssim_values.append(ssim)

    print(f"{filename} - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

# Calculate average metrics
avg_psnr = np.mean(psnr_values) if psnr_values else 0
avg_ssim = np.mean(ssim_values) if ssim_values else 0

print("\nOverall Metrics:")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")

