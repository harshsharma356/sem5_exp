import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

otsu_value, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

mean_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

gaussian_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(global_thresh, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(otsu_thresh, cmap="gray")
plt.title(f"Otsu Thresholding (T={otsu_value:.0f})")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(mean_thresh, cmap="gray")
plt.title("Mean Adaptive")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(gaussian_thresh, cmap="gray")
plt.title("Gaussian Adaptive")
plt.axis("off")

plt.tight_layout()
plt.show()