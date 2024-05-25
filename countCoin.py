import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('coin8.jpg') # gambar 6 & 4 bertumpuk

def adjust_brightness(image, brightness_factor):
    image = image.astype(np.float32)
    bright_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    return bright_image

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def adjust_contrast(image, contrast_factor):
    image = image.astype(np.float32)
    mean = np.mean(image)
    contrast_image = mean + contrast_factor * (image - mean)
    contrast_image = np.clip(contrast_image, 0, 255).astype(np.uint8)
    return contrast_image

# Sharpen and adjust the brightness, contrast, & sharpen of the image
contrast_factor = 1

image_sharp = sharpen_image(img)
image_bright = adjust_brightness(image_sharp, 3)
image_contrast = adjust_contrast(image_bright, contrast_factor)

# Convert the image to grayscale
image_gray = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
# convert to black & white
(thresh, blackAndWhiteImage) = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

# clear noise
bilateral = cv2.bilateralFilter(image_gray, 5, 100, 100)

# Apply Canny edge detection
canny = cv2.Canny(bilateral, 100, 200)

# Dilate the edges to close gaps
dilated = cv2.dilate(canny, (3, 3), iterations=2)

# Find contours
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours
min_area = 500  # Adjust this value to set the minimum area for a valid contour
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

# Count the number of coins
num_coins = len(filtered_contours)

# Convert the Canny image to RGB for visualization
rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
cv2.drawContours(rgb, filtered_contours, -1, (0, 255, 0), 2)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.suptitle('21104410049 - Viery Nugroho', fontsize=16, fontweight='bold')
plt.title('Coin Detection', fontsize=14, fontweight='bold')

# Original image
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(img)
plt.axis('off')

# Canny edges with count
plt.subplot(1, 2, 2)
plt.title(f'Canny - Number of coins: {num_coins}')
plt.imshow(rgb)
plt.axis('off')

plt.show()

# Save the result
# cv2.imwrite('coin_detection_result.jpg', rgb)