import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def roberts_cross(img_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gx = np.array([[1, 0], [0, -1]])
    gy = np.array([[0, 1], [-1, 0]])

    gradient_x = ndimage.convolve(image, gx)
    gradient_y = ndimage.convolve(image, gy)

    magnitude = np.sqrt(gradient_x * 2 + gradient_y * 2)

    # 3 variasi - 5 10 50
    threshold = 50
    edges = magnitude > threshold

    return edges



image_path = 'doraemon.png'
edge_image = roberts_cross(image_path)


original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
# Display the original image using matplotlib
plt.imshow(original_image_rgb)
plt.title('Original Image')
plt.axis('off')

# Edge-detected Image
plt.subplot(1, 2, 2)
# Display the edge-detected image using matplotlib with a grayscale color map.
plt.imshow(edge_image, cmap='gray')
plt.title('Edge-detected Image')
plt.axis('off')

# Show the plot containing both images.
plt.show()