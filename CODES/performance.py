import cv2
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Read the original image
original_image_path = r'C:\Users\C RISHI VARDHAN REDD\Desktop\osteoathritis\test\4\9048789L.png'  # Replace with the path to your original image
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Error: Unable to load the image. Check the file path.")
    exit()

# Step 1: Gaussian Median Noise Cancellation (Gaussian Blur)
gaussian_blur = cv2.GaussianBlur(original_image, (5, 5), 0)

# Step 2: Histogram Equalization
hist_eq = cv2.equalizeHist(gaussian_blur)

# Step 3: Unsharp Masking
blurred = cv2.GaussianBlur(hist_eq, (5, 5), 0)
unsharp_masked = cv2.addWeighted(hist_eq, 1.5, blurred, -0.5, 0)

# Calculate Mean Squared Error (MSE)
def mse(image1, image2):
    return np.sum((image1 - image2) ** 2) / float(image1.shape[0] * image1.shape[1])

# Calculate Peak Signal-to-Noise Ratio (PSNR)
def psnr(image1, image2):
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return 100  # PSNR is infinite if MSE is zero
    return 10 * np.log10((255 ** 2) / mse_value)

# Calculate Structural Similarity Index (SSIM)
def ssim_index(image1, image2):
    return ssim(image1, image2)

# Calculate metrics for denoised images
psnr_denoised = psnr(original_image, unsharp_masked)
ssim_denoised = ssim_index(original_image, unsharp_masked)
mse_denoised = mse(original_image, unsharp_masked)

# Display results
print(f"PSNR (Denoised Image): {psnr_denoised:.2f} dB")
print(f"SSIM (Denoised Image): {ssim_denoised:.4f}")
print(f"MSE (Denoised Image): {mse_denoised:.2f}")

# Optionally, plot the images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(gaussian_blur, cmap='gray')
axes[1].set_title("Gaussian Median Noise Cancel")
axes[1].axis('off')

axes[2].imshow(hist_eq, cmap='gray')
axes[2].set_title("Histogram Equalized")
axes[2].axis('off')

axes[3].imshow(unsharp_masked, cmap='gray')
axes[3].set_title("Unsharp Masking")
axes[3].axis('off')

plt.show()
