import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
cx = w / 2
cy = h / 2
angle = 30
theta = np.deg2rad(angle)
alpha = np.cos(theta)
beta = np.sin(theta)
M_manual = np.array([[alpha, beta, (1 - alpha) * cx - beta * cy], [-beta, alpha, beta * cx + (1 - alpha) * cy]], dtype=np.float32)
manual_rotation = cv2.warpAffine(gray, M_manual, (w, h))
M_opencv = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
opencv_rotation = cv2.warpAffine(gray, M_opencv, (w, h))
difference = cv2.absdiff(manual_rotation, opencv_rotation)
print("Manual Matrix:")
print(M_manual)
print("OpenCV Matrix:")
print(M_opencv)
print("Maximum Pixel Difference:", np.max(difference))
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(manual_rotation, cmap="gray")
plt.title("Manual Rotation")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(opencv_rotation, cmap="gray")
plt.title("OpenCV Rotation")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(difference, cmap="gray")
plt.title("Difference")
plt.axis("off")
plt.tight_layout()
plt.show()