import cv2
import numpy as np

# Load the original and template images
img1 = cv2.imread('original.png', 1)
template = cv2.imread('template.png', 0)

# Convert the images to grayscale
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Get the dimensions of the template
w, h = template.shape[::-1]

# Perform template matching using different methods
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2=img1.copy()
    # Apply template matching
    result = cv2.matchTemplate(gray_img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # Choose the appropriate threshold based on the method
    if method in [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
        threshold = max_val-0.05
        print(threshold)
        # Find all occurrences above the threshold
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            # Draw a rectangle around the matched region
            cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    else:
        print(threshold)
        threshold = min_val*2
        loc = np.where(result <= threshold)
        for pt in zip(*loc[::-1]):
            # Draw a rectangle around the matched region
            cv2.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    # Display the matched image
    cv2.imshow('Matched Image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
