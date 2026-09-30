import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
cx = w / 2
cy = h / 2
angle = np.deg2rad(20)
R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
T_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
T_back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
R_center = T_back @ R @ T_center
T = np.array([[1, 0, 20], [0, 1, 20], [0, 0, 1]], dtype=np.float32)
S = np.array([[0.75, 0, 0], [0, 0.75, 0], [0, 0, 1]], dtype=np.float32)
M = S @ T @ R_center
composed = cv2.warpAffine(gray, M[:2], (w, h))
rotation_matrix = cv2.getRotationMatrix2D((cx, cy), 20, 1.0)
rotated = cv2.warpAffine(gray, rotation_matrix, (w, h))
translation_matrix = np.float32([[1, 0, 20], [0, 1, 20]])
translated = cv2.warpAffine(rotated, translation_matrix, (w, h))
scaling_matrix = np.float32([[0.75, 0, 0], [0, 0.75, 0]])
sequential = cv2.warpAffine(translated, scaling_matrix, (w, h))
difference = cv2.absdiff(composed, sequential)
print("Composed Affine Matrix:")
print(M[:2])
print("Maximum Pixel Difference:", np.max(difference))
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(composed, cmap="gray")
plt.title("Composed Transformation")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(sequential, cmap="gray")
plt.title("Sequential Transformation")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(difference, cmap="gray")
plt.title("Difference")
plt.axis("off")
plt.tight_layout()
plt.show()