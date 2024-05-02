import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the stereo pair of images
left_image = cv2.imread('left.png', 0)  #left image path
right_image = cv2.imread('right.png', 0)  #right image path
print(left_image.shape)
print(right_image.shape)
left_image=cv2.resize(left_image,(400,300 ))
right_image=cv2.resize(right_image,(400,300 ))

# Create a StereoBM object
stereo = cv2.StereoBM_create(numDisparities =16,blockSize = 15)  # blockSize reduced to 11

# Compute the disparity map
disparity_map = stereo.compute(left_image, right_image)

# Create a figure with 3 subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Display the original image
axs[0].imshow(left_image, cmap='gray')
axs[0].set_title('left Image')
axs[1].imshow(right_image, cmap='gray')
axs[1].set_title('Right Image')

# Display the disparity map
axs[2].imshow(disparity_map, cmap='gray')
axs[2].set_title('Disparity Map')
plt.show()
