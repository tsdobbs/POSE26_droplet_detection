# This file is a collection of sandbox scripts for trying to analyze droplet images.
# Nothing works great yet!


########################################################################
# Use HoughCircles method to detect circles and output an image displaying these circles.
# Works okay at detecting droplets, but is probably not the right approach.
########################################################################

import cv2 as cv
import numpy as np

image_path = "./images/Scan10006.jpg"
output_path = "./output/Scan10006.png"

img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 300, param1=100, param2=30, minRadius=10, maxRadius=400
)
assert circles is not None, "No circles detected"

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # img, (x,y), r, color, ??
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imwrite(output_path, cimg)


########################################################################
# Detect contours and write data to an image file and csv with area, circularity,
# and position data
########################################################################

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path

input_path = "./images/Scan10006.jpg"
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Could not read {input_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
rows = []

i = 1
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 20:
        continue

    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
    cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(
        vis,
        str(i),
        (cx + 5, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
    )

    rows.append(
        {
            "id": i,
            "area": area,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
        }
    )
    i += 1

out_img = output_dir / "contours.png"
out_csv = output_dir / "contours.csv"

cv2.imwrite(str(out_img), vis)
pd.DataFrame(rows).to_csv(out_csv, index=False)

with open(str(out_img) + ".meta.json", "w") as f:
    json.dump(
        {
            "caption": "Detected droplet contours",
            "description": "Contour outlines overlaid on the original image, with each droplet labeled.",
        },
        f,
    )

print(f"Saved: {out_img}")
print(f"Saved: {out_csv}")


########################################################################
# Try again with more documentation
########################################################################

import cv2 as cv
import numpy as np
import pandas as pd

# -----------------------
# User-adjustable params
# -----------------------
IMAGE_PATH = "./images/playground_V2.png"

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
THRESH_VALUE = 190  # for global thresholding
MAX_VALUE = 255
ADAPTIVE_BLOCK_SIZE = 31  # must be odd
ADAPTIVE_C = 5
INVERT = False  # set True if droplets are darker than background

# Morphology
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 3

# Blob filtering
MIN_AREA = 50
MAX_AREA = 1000
MIN_CIRCULARITY = 0.5
MAX_CIRCULARITY = 1.2

# Output
CSV_OUT = "output/droplet_measurements.csv"

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

    rows.append(
        {
            "label": label,
            "area_px": contour_area,
            "perimeter_px": perimeter,
            "circularity": circularity,
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
        }
    )

    cv.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    cv.circle(overlay, (int(cx), int(cy)), 3, (0, 0, 255), -1)

df = pd.DataFrame(rows).sort_values(["centroid_y", "centroid_x"]).reset_index(drop=True)
df.to_csv(CSV_OUT, index=False)

print(f"Detected droplets: {len(df)}")
print(df[["area_px", "circularity"]].head())

cv.imwrite("output/overlay_detections.png", overlay)
cv.imwrite("output/mask.png", mask)
