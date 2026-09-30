import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

h, w = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
rotated = cv2.warpAffine(img, rotation_matrix, (w, h))

translation_matrix = cv2.getRotationMatrix2D((0, 0), 0, 1)
translation_matrix[0, 2] = 100
translation_matrix[1, 2] = 50
translated = cv2.warpAffine(img, translation_matrix, (w, h))

scaled = cv2.resize(img, None, fx=0.5, fy=0.5)

flipped_horizontal = cv2.flip(img, 1)
flipped_vertical = cv2.flip(img, 0)

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(rotated)
plt.title("Rotation 45°")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(translated)
plt.title("Translation")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(scaled)
plt.title("Scaling 0.5")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(flipped_horizontal)
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(flipped_vertical)
plt.title("Vertical Flip")
plt.axis("off")

plt.tight_layout()
plt.show()