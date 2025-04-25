import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image_path = r'C:\Users\C RISHI VARDHAN REDD\Desktop\osteoathritis\test\4\9048789L.png'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load the image. Check the file path.")
    exit()

# Step 1: Gaussian Median Noise Cancellation (Gaussian Blur)
# Apply Gaussian blur to reduce noise
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Histogram Equalization
# Apply histogram equalization to enhance image contrast
hist_eq = cv2.equalizeHist(gaussian_blur)

# Step 3: Unsharp Masking
# Perform unsharp masking to enhance the image details
blurred = cv2.GaussianBlur(hist_eq, (5, 5), 0)
unsharp_masked = cv2.addWeighted(hist_eq, 1.5, blurred, -0.5, 0)

# Plot the images after each technique
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
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
