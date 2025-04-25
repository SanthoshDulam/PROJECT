import cv2
import numpy as np

# Use raw string or double backslashes to avoid path issues
image = cv2.imread(r'C:\Users\C RISHI VARDHAN REDD\Desktop\osteoathritis\auto_test\0\9003175_1.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error: Unable to load the image. Check the file path.")
else:
    # Check image properties
    min_value = np.min(image)   # Minimum pixel value
    max_value = np.max(image)   # Maximum pixel value
    image_shape = image.shape   # Image dimensions

    # Display the properties
    print("Image Shape:", image_shape)
    print("Minimum Value:", min_value)
    print("Maximum Value:", max_value)
    print("Range:", max_value - min_value)

    # Optionally, show the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

