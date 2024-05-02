import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('cell.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to be a list of RGB values
pixels = image.reshape(-1, 3)

# Perform K-Means
kmeans = KMeans(n_clusters=2)  # Change the number of clusters as needed
kmeans.fit(pixels)

# Replace each pixel with its centroid
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

# Convert data types for imshow
image = image.astype(np.uint8)
segmented_img = segmented_img.astype(np.uint8)

# Create a subplot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Show original image
ax[0].imshow(image)
ax[0].set_title('Original Image')

# Show segmented image
ax[1].imshow(segmented_img)
ax[1].set_title('Segmented Image')

plt.show()
