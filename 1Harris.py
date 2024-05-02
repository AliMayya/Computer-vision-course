import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image=cv2.imread("box.jpg",1)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 5, 0.04)

# Dilate the result for better visualization
dst = cv2.dilate(dst, None)

# Threshold for corner points
threshold = 0.01 * dst.max()
image[dst > threshold] = [0, 0, 255]  # Mark corners in red

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")
plt.show()
