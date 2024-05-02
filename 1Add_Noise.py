import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Panda.jpg', cv2.IMREAD_GRAYSCALE)
noise_types = ['gaussian', 'salt & pepper', 'gamma', 'speckle']
noisy_imgs = []
for noise_type in noise_types:
    if noise_type == 'gaussian':
        # Add true Gaussian noise
        noise = np.random.normal(0,0.5,img.shape).astype('uint8')
        noisy_img=cv2.add(img,noise)
        noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values are within [0, 255]
    elif noise_type == 'salt & pepper':
        noisy_img = img.copy()
        noise = np.zeros_like(img)
        cv2.randu(noise, 0, 255)
        noisy_img[noise < 10] = 0
        noisy_img[noise > 245] = 255
    elif noise_type == 'gamma':
        noisy_img = np.power(img /np.max(img), 1.5) * 255
        noisy_img = noisy_img.astype(np.uint8)
    elif noise_type == 'uniform':
        noisy_img = img.copy()
        noise = np.random.uniform(-50, 50, size=img.shape)
        noisy_img = noisy_img + noise
        noisy_img[noisy_img < 0] = 0
        noisy_img[noisy_img > 255] = 255    
    elif noise_type == 'speckle':
        noisy_img = img + img * np.random.randn(*img.shape) * 0.1
    noisy_imgs.append(noisy_img)    

# Create a single subplot for all images
fig, axs = plt.subplots(2, len(noise_types) + 1, figsize=(15, 6))
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

for i, noise_type in enumerate(noise_types):
    axs[0, i + 1].imshow(noisy_imgs[i], cmap='gray')
    axs[0, i + 1].set_title(noise_type)
    axs[0, i + 1].axis('off')

axs[1, 0].hist(img.ravel(), 256, [0, 256], alpha=0.5, color='red')
axs[1, 0].set_title(' Original Histogram')
# Plot histograms
for i, noise_type in enumerate(noise_types):
    axs[1, i + 1].hist(noisy_imgs[i].ravel(), 256, [0, 256], alpha=0.5, color='red')
    axs[1, i + 1].set_title(f'{noise_type} Histogram')

plt.tight_layout()
plt.show()
