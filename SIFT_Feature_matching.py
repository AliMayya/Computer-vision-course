import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the images
original = cv2.imread('original.jpg', 1)  # Original image
template = cv2.imread('template1.jpg', 1)  # Template image

# Create SIFT object
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT for both images
kp1, des1 = sift.detectAndCompute(original, None)
kp2, des2 = sift.detectAndCompute(template, None)

# Draw keypoints on the images
original_keypoints = cv2.drawKeypoints(original, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
template_keypoints = cv2.drawKeypoints(template, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the images with keypoints
cv2.imshow('Original Image with Keypoints', original_keypoints)
cv2.imshow('Template Image with Keypoints', template_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

# FLANN parameters for feature matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
search_params = dict(checks=5)

flann = cv2.FlannBasedMatcher(index_params, search_params)
#flann= cv2.BFMatcher()
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches (ratio test as per Lowe's paper)
good_matches = []
for m, n in matches:
    if m.distance < 0.7* n.distance:
        good_matches.append(m)

MIN_MATCH_COUNT = 35  # Adjust this threshold as needed

if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate the transformation matrix using RANSAC
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply the transformation to the template corners
    h, w, ss = template.shape
    template_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    transformed_corners = cv2.perspectiveTransform(template_corners.reshape(1, -1, 2), M)

    # Draw the bounding box around the detected template
    original_with_box = cv2.polylines(original, [np.int32(transformed_corners)], True, (255, 0, 0), 3)
    
    # Show the result
    cv2.imshow('Detected Template in Original Image', original_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img3=cv2.drawMatchesKnn(original, kp1, template, kp2, [good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT))
