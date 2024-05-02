import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('rose.jpg',1)

# Convert the image from BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for red color in HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower_red, upper_red)

# Define range for red color in HSV
lower_red = np.array([170, 120, 70])
upper_red = np.array([180, 255, 255])

# Threshold the HSV image to get only red colors
mask2 = cv2.inRange(hsv, lower_red, upper_red)

# Combine the two masks
mask = mask1 + mask2

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

# Convert the BGR images to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

# Create subplots
fig, axs = plt.subplots(2, 1)

# Show the original image
axs[0].imshow(img_rgb)
axs[0].set_title('Original Image')

# Show the result image
axs[1].imshow(res_rgb)
axs[1].set_title('Segmented Red Color')

# Display the plot
plt.show()

