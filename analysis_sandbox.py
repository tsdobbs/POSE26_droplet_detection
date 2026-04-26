# This file contains a script to analyze a png or jpg image for droplet detection.
# It outputs images that reflect the detection as well as a CSV with droplet
# location, area, perimeter, and circularity.

import cv2 as cv
import numpy as np
import pandas as pd

from droplet_content_analysis import get_luminance

# Filepaths
IMAGE_PATH = "./images/playground_V2.png"
CSV_OUT = "output/droplet_measurements.csv"
OVERLAY_OUT = "output/overlay_detections.png"
MASK_OUT = "output/mask.png"

# Contrast enhancement
USE_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Optional brightness/contrast tuning: out = alpha * img + beta
ALPHA = 1.0  # contrast gain
BETA = 0.0  # brightness shift

# Denoising
BLUR_METHOD = "gaussian"  # "gaussian" or "median"
GAUSSIAN_KSIZE = (11, 11)  # must be odd numbers
MEDIAN_KSIZE = 5

# Thresholding
USE_ADAPTIVE = False
THRESH_VALUE = 210  # for global thresholding
MAX_VALUE = 255
ADAPTIVE_BLOCK_SIZE = 31  # must be odd
ADAPTIVE_C = 5
INVERT = False  # set True if droplets are darker than background

# Morphology
MORPH_KERNEL_SIZE = 5
MORPH_OPEN_ITER = 0
MORPH_CLOSE_ITER = 4

# Blob filtering
MIN_AREA = 50
MAX_AREA = 1000
MIN_CIRCULARITY = 0.5
MAX_CIRCULARITY = 1.2

### Load and preprocess
img = cv.imread(IMAGE_PATH, cv.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Could not read {IMAGE_PATH}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Step 1: contrast enhancement
if USE_CLAHE:
    clahe = cv.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    gray = clahe.apply(gray)

gray = cv.convertScaleAbs(gray, alpha=ALPHA, beta=BETA)

# Step 2: denoise
if BLUR_METHOD == "gaussian":
    gray_blur = cv.GaussianBlur(gray, GAUSSIAN_KSIZE, 0)
elif BLUR_METHOD == "median":
    gray_blur = cv.medianBlur(gray, MEDIAN_KSIZE)
else:
    gray_blur = gray.copy()

# Step 3: threshold
if USE_ADAPTIVE:
    thresh_type = cv.THRESH_BINARY_INV if INVERT else cv.THRESH_BINARY
    mask = cv.adaptiveThreshold(
        gray_blur,
        MAX_VALUE,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )
else:
    thresh_type = cv.THRESH_BINARY_INV if INVERT else cv.THRESH_BINARY
    _, mask = cv.threshold(gray_blur, THRESH_VALUE, MAX_VALUE, thresh_type)

# Step 4: morphology cleanup
kernel = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITER)


### Blob detection
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
    mask, connectivity=8
)

rows = []
overlay = img.copy()

for label in range(1, num_labels):  # skip background
    area = int(stats[label, cv.CC_STAT_AREA])
    if area < MIN_AREA or area > MAX_AREA:
        continue

    component_mask = (labels == label).astype(np.uint8) * 255
    contours, _ = cv.findContours(
        component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        continue

    contour = max(contours, key=cv.contourArea)
    contour_area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if perimeter <= 0:
        continue

    circularity = 4 * np.pi * contour_area / (perimeter * perimeter)

    if circularity < MIN_CIRCULARITY or circularity > MAX_CIRCULARITY:
        continue

    x, y, w, h = (
        stats[label, cv.CC_STAT_LEFT],
        stats[label, cv.CC_STAT_TOP],
        stats[label, cv.CC_STAT_WIDTH],
        stats[label, cv.CC_STAT_HEIGHT],
    )
    cx, cy = centroids[label]

    luminance = get_luminance(IMAGE_PATH, cx, cy)

    rows.append(
        {
            "label": label,
            "filename": CSV_OUT,
            "area_px": contour_area,
            "perimeter_px": perimeter,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "luminance": luminance
        }
    )

    cv.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv.circle(overlay, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv.putText(overlay, str(int(luminance)), (int(cx) + 200, int(cy)), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255,255,255), thickness=2)

df = pd.DataFrame(rows).sort_values(["centroid_y", "centroid_x"]).reset_index(drop=True)
df.to_csv(CSV_OUT, index=False)

print(f"Detected droplets: {len(df)}")
print(df[["area_px", "circularity"]].head())

cv.imwrite(OVERLAY_OUT, overlay)
cv.imwrite(MASK_OUT, mask)
