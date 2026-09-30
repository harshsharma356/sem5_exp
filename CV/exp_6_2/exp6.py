import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ============================================================
# HELPER FUNCTIONS
# ============================================================

own_image_path = "/media/galdrux/galdrux_storage/sem_5/CV/dataset/satellite_9.jpg"

def show_image_hist(img, title, color=False):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    if color:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    else:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
    plt.title(title)
    plt.subplot(1,2,2)
    if color:
        colors = ("b", "g", "r")
        for i, c in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0,256])
            plt.plot(hist, label=c.upper())
        plt.legend()
    else:
        hist = cv2.calcHist([img], [0], None, [256], [0,256])
        plt.plot(hist)
    plt.xlim([0,255])
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.tight_layout()
    plt.show()

def print_stats(img, name):
    print(name)
    print("Minimum :", np.min(img))
    print("Maximum :", np.max(img))
    print("Mean    :", np.mean(img))
    print("Std Dev :", np.std(img))
    print()

def plot_histogram(img, title):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    plt.figure(figsize=(8,4))
    plt.plot(hist)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.xlim([0,255])
    plt.grid()
    plt.show()

def plot_normalized_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    total_pixels = img.shape[0] * img.shape[1]
    normalized_hist = hist.flatten() / total_pixels
    print("Total pixel count:", total_pixels)
    print("Sum of normalized histogram:", np.sum(normalized_hist))
    plt.figure(figsize=(8,4))
    plt.plot(normalized_hist)
    plt.title("Normalized Histogram p(r)")
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.xlim([0,255])
    plt.grid()
    plt.show()
    return hist.flatten(), normalized_hist

def plot_cdf(normalized_hist):
    cdf = np.cumsum(normalized_hist)
    plt.figure(figsize=(8,4))
    plt.plot(cdf)
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Intensity")
    plt.ylabel("CDF")
    plt.xlim([0,255])
    plt.ylim([0,1.05])
    plt.grid()
    plt.show()
    return cdf


# ============================================================
# A1 - MAKE A LOW-CONTRAST IMAGE
# ============================================================

camera = data.camera()
camera = camera.astype(np.uint8)

low_contrast = cv2.normalize(
    camera,
    None,
    alpha=90,
    beta=160,
    norm_type=cv2.NORM_MINMAX
)

print("A1: Original Camera Image")
show_image_hist(camera, "Original Camera Image")
print_stats(camera, "Original Camera Image")

print("A1: Low-Contrast Image")
show_image_hist(low_contrast, "Low-Contrast Camera Image")
print_stats(low_contrast, "Low-Contrast Camera Image")


# ============================================================
# A2 - HISTOGRAM, NORMALIZED HISTOGRAM AND CDF
# ============================================================

print("A2: Histogram")

hist, normalized_hist = plot_normalized_histogram(low_contrast)

print("A2: CDF")

cdf = plot_cdf(normalized_hist)


# ============================================================
# A3 - CONTRAST STRETCHING
# ============================================================

print("A3: Contrast Stretching")

r_min = np.min(low_contrast)
r_max = np.max(low_contrast)

stretched_manual = (
    (low_contrast.astype(np.float32) - r_min)
    / (r_max - r_min) * 255
).astype(np.uint8)

stretched_cv = cv2.normalize(
    low_contrast,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

difference_stretching = np.max(
    cv2.absdiff(stretched_manual, stretched_cv)
)

print("Minimum intensity:", r_min)
print("Maximum intensity:", r_max)
print("Maximum difference between manual and cv2.normalize:",
      difference_stretching)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(low_contrast, cmap="gray")
plt.title("Before Contrast Stretching")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(stretched_manual, cmap="gray")
plt.title("After Contrast Stretching")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(cv2.calcHist([low_contrast],[0],None,[256],[0,256]))
plt.title("Before Histogram")
plt.xlim([0,255])

plt.subplot(2,2,4)
plt.plot(cv2.calcHist([stretched_manual],[0],None,[256],[0,256]))
plt.title("After Histogram")
plt.xlim([0,255])

plt.tight_layout()
plt.show()

print_stats(low_contrast, "Low-Contrast Image")
print_stats(stretched_manual, "Contrast-Stretched Image")


# ============================================================
# A4 - GLOBAL HISTOGRAM EQUALIZATION
# ============================================================

print("A4: Global Histogram Equalization")

global_equalized = cv2.equalizeHist(low_contrast)

show_image_hist(
    global_equalized,
    "Global Histogram Equalization"
)

print_stats(global_equalized, "Globally Equalized Image")


# ============================================================
# A5 - MANUAL HISTOGRAM EQUALIZATION
# ============================================================

print("A5: Manual Histogram Equalization")

hist = cv2.calcHist(
    [low_contrast],
    [0],
    None,
    [256],
    [0,256]
).flatten()

total_pixels = low_contrast.size

probability = hist / total_pixels

cdf = np.cumsum(probability)

cdf_min = cdf[np.nonzero(cdf)][0]

manual_lut = np.round(
    (cdf - cdf_min) /
    (1 - cdf_min) * 255
)

manual_lut[cdf <= cdf_min] = 0

manual_lut = np.clip(
    manual_lut,
    0,
    255
).astype(np.uint8)

manual_equalized = manual_lut[low_contrast]

max_difference = np.max(
    cv2.absdiff(
        manual_equalized,
        global_equalized
    )
)

print("Maximum difference between manual and cv2.equalizeHist:",
      max_difference)

plt.figure(figsize=(8,4))
plt.plot(manual_lut)
plt.title("T(r) Transfer Function")
plt.xlabel("Input Intensity r")
plt.ylabel("Output Intensity T(r)")
plt.xlim([0,255])
plt.ylim([0,255])
plt.grid()
plt.show()

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.imshow(global_equalized, cmap="gray")
plt.title("OpenCV Equalization")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(manual_equalized, cmap="gray")
plt.title("Manual Equalization")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# A6 - CLAHE
# ============================================================

print("A6: CLAHE")

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

clahe_result = clahe.apply(low_contrast)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(global_equalized, cmap="gray")
plt.title("Global Equalization")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(clahe_result, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(
    cv2.calcHist(
        [global_equalized],
        [0],
        None,
        [256],
        [0,256]
    )
)
plt.title("Global Equalization Histogram")
plt.xlim([0,255])

plt.subplot(2,2,4)
plt.plot(
    cv2.calcHist(
        [clahe_result],
        [0],
        None,
        [256],
        [0,256]
    )
)
plt.title("CLAHE Histogram")
plt.xlim([0,255])

plt.tight_layout()
plt.show()

print_stats(global_equalized, "Global Equalization")
print_stats(clahe_result, "CLAHE")


# ============================================================
# A7 - CLAHE PARAMETER VARIATION
# ============================================================

print("A7: CLAHE Parameter Variation")

clip_limits = [1.0, 2.0, 8.0]

tile_sizes = [
    (4,4),
    (8,8),
    (16,16)
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(12,12)
)

for i, clip in enumerate(clip_limits):
    for j, tile in enumerate(tile_sizes):

        clahe_test = cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=tile
        )

        result = clahe_test.apply(low_contrast)

        axes[i,j].imshow(
            result,
            cmap="gray"
        )

        axes[i,j].set_title(
            f"Clip={clip}, Tile={tile[0]}x{tile[1]}"
        )

        axes[i,j].axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# A8 - COLOUR IMAGE PROCESSING
# ============================================================

print("A8: Colour Image Processing")

colour_img = cv2.imread(
    own_image_path
)

if colour_img is None:
    raise FileNotFoundError(
        f"Could not load {own_image_path}"
    )


# ------------------------------------------------------------
# A8.1 - Y-CHANNEL EQUALIZATION
# ------------------------------------------------------------

ycrcb = cv2.cvtColor(
    colour_img,
    cv2.COLOR_BGR2YCrCb
)

Y, Cr, Cb = cv2.split(
    ycrcb
)

Y_equalized = cv2.equalizeHist(
    Y
)

ycrcb_equalized = cv2.merge(
    [Y_equalized, Cr, Cb]
)

y_equalized_colour = cv2.cvtColor(
    ycrcb_equalized,
    cv2.COLOR_YCrCb2BGR
)


# ------------------------------------------------------------
# A8.2 - INDEPENDENT RGB EQUALIZATION
# ------------------------------------------------------------

B, G, R = cv2.split(
    colour_img
)

B_equalized = cv2.equalizeHist(B)

G_equalized = cv2.equalizeHist(G)

R_equalized = cv2.equalizeHist(R)

rgb_equalized = cv2.merge(
    [
        B_equalized,
        G_equalized,
        R_equalized
    ]
)


# ------------------------------------------------------------
# A8.3 - COLOUR COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(
    cv2.cvtColor(
        colour_img,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Original Colour Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(
    cv2.cvtColor(
        y_equalized_colour,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Y-Channel Equalization")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(
    cv2.cvtColor(
        rgb_equalized,
        cv2.COLOR_BGR2RGB
    )
)
plt.title("Independent RGB Equalization")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# PART C - OWN PHOTOGRAPH
# ============================================================

print("PART C: OWN IMAGE")


# ------------------------------------------------------------
# C1 - LOAD AND CONVERT OWN IMAGE TO GRAYSCALE
# ------------------------------------------------------------

own_gray = cv2.cvtColor(
    colour_img,
    cv2.COLOR_BGR2GRAY
)

print_stats(
    own_gray,
    "Original Own Image - Grayscale"
)


# ------------------------------------------------------------
# C2 - HISTOGRAM, NORMALIZED HISTOGRAM AND CDF
# ------------------------------------------------------------

plot_histogram(
    own_gray,
    "Own Image Histogram"
)

hist_own = cv2.calcHist(
    [own_gray],
    [0],
    None,
    [256],
    [0,256]
).flatten()

normalized_own = (
    hist_own /
    own_gray.size
)

cdf_own = np.cumsum(
    normalized_own
)

print("Total pixel count:", own_gray.size)

print(
    "Sum of normalized histogram:",
    np.sum(normalized_own)
)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(hist_own)
plt.title("Own Image h(r)")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.xlim([0,255])

plt.subplot(1,3,2)
plt.plot(normalized_own)
plt.title("Own Image p(r)")
plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.xlim([0,255])

plt.subplot(1,3,3)
plt.plot(cdf_own)
plt.title("Own Image CDF")
plt.xlabel("Intensity")
plt.ylabel("CDF")
plt.xlim([0,255])
plt.ylim([0,1.05])

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# C3 - CONTRAST STRETCHING ON OWN IMAGE
# ------------------------------------------------------------

print("Own Image: Contrast Stretching")

own_min = np.min(
    own_gray
)

own_max = np.max(
    own_gray
)

own_stretched = (
    (own_gray.astype(np.float32) - own_min)
    / (own_max - own_min) * 255
).astype(np.uint8)

own_stretched_cv = cv2.normalize(
    own_gray,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

own_stretch_difference = np.max(
    cv2.absdiff(
        own_stretched,
        own_stretched_cv
    )
)

print(
    "Maximum difference between manual and cv2.normalize:",
    own_stretch_difference
)

show_image_hist(
    own_stretched,
    "Own Image - Contrast Stretched"
)

print_stats(
    own_stretched,
    "Own Image - Contrast Stretched"
)


# ------------------------------------------------------------
# C4 - GLOBAL HISTOGRAM EQUALIZATION ON OWN IMAGE
# ------------------------------------------------------------

print("Own Image: Global Histogram Equalization")

own_global = cv2.equalizeHist(
    own_gray
)

show_image_hist(
    own_global,
    "Own Image - Global Equalization"
)

print_stats(
    own_global,
    "Own Image - Global Equalization"
)


# ------------------------------------------------------------
# C5 - CLAHE ON OWN IMAGE
# ------------------------------------------------------------

print("Own Image: CLAHE")

own_clahe_object = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

own_clahe = own_clahe_object.apply(
    own_gray
)

show_image_hist(
    own_clahe,
    "Own Image - CLAHE"
)

print_stats(
    own_clahe,
    "Own Image - CLAHE"
)


# ------------------------------------------------------------
# C6 - OWN IMAGE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(
    own_gray,
    cmap="gray"
)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(
    own_global,
    cmap="gray"
)
plt.title("Global Equalization")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(
    own_clahe,
    cmap="gray"
)
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# PART B - FINAL MEASUREMENT TABLE FOR OWN IMAGE
# ============================================================

print("FINAL MEASUREMENT TABLE")

results = {
    "Original Own Image": own_gray,
    "Contrast Stretched": own_stretched,
    "Globally Equalized": own_global,
    "CLAHE (2.0, 8x8)": own_clahe
}

print(
    f"{'Image':<25}"
    f"{'Min':>8}"
    f"{'Max':>8}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
)

print("-" * 65)

for name, image in results.items():

    print(
        f"{name:<25}"
        f"{np.min(image):>8}"
        f"{np.max(image):>8}"
        f"{np.mean(image):>12.2f}"
        f"{np.std(image):>12.2f}"
    )