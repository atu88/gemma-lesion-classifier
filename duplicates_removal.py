import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import shannon_entropy
from shutil import move

# Parameters: Change them accordingly
image_folder = 'images' 
bad_folder = 'duplicates'
metadata_file = 'bcn20000_metadata.csv'

# Scoring weights
w_blur = 0.6
w_entropy = -0.3
w_edges = -0.4

# Ensure duplicates folder exists
os.makedirs(bad_folder, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_file)

# Scoring Functions
def get_blur(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_entropy(image_path):
    try:
        img = imread(image_path, as_gray=True)
        return shannon_entropy(img)
    except:
        return 0

def get_edge_density(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    edges = cv2.Canny(img, 50, 150)
    return np.sum(edges > 0) / edges.size

def has_dark_border(image_path, threshold=30, dark_pixel_ratio=0.4):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = min(h, w) // 2 - 5
    center = (w // 2, h // 2)
    cv2.circle(mask, center, radius, 255, -1)
    border_mask = cv2.bitwise_not(mask)
    border_pixels = img[border_mask > 0]
    dark_pixels = np.sum(border_pixels < threshold)
    ratio = dark_pixels / len(border_pixels)
    return ratio > dark_pixel_ratio

def compute_score(image_path):
    blur = get_blur(image_path)
    entropy = get_entropy(image_path)
    edge_density = get_edge_density(image_path)
    score = (w_blur * blur) + (w_entropy * entropy) + (w_edges * edge_density)
    return score, blur, entropy, edge_density

# Main Loop
best_images = []
rejected_images = []

for lesion_id, group in df.groupby('lesion_id'):
    if len(group) == 1:
        best_images.append(group.iloc[0])
        continue

    scored_rows = []
    for _, row in group.iterrows():
        isic_id = row['isic_id']
        img_path = os.path.join(image_folder, f"{isic_id}.jpg")
        if os.path.exists(img_path):
            score, blur, entropy, edge_density = compute_score(img_path)
            dark_circle = has_dark_border(img_path)
            scored_rows.append((row, score, isic_id, dark_circle))

    if not scored_rows:
        continue

    # Prefer images WITHOUT dark borders if mixed group
    has_mix = any(x[3] for x in scored_rows) and any(not x[3] for x in scored_rows)
    if has_mix:
        scored_rows.sort(key=lambda x: (x[3], -x[1]))  # dark_circle first (True > False), then score
    else:
        scored_rows.sort(key=lambda x: -x[1])  # by score only

    # Keep best
    best_row = scored_rows[0]
    best_images.append(best_row[0])

    # Move the others
    for row, score, isic_id, _ in scored_rows[1:]:
        src = os.path.join(image_folder, f"{isic_id}.jpg")
        dst = os.path.join(bad_folder, f"{isic_id}.jpg")
        if os.path.exists(src):
            move(src, dst)
            rejected_images.append(isic_id)

# Save new metadata
pd.DataFrame(best_images).to_csv('clean_metadata_bcn.csv', index=False)
pd.DataFrame({'rejected_isic_id': rejected_images}).to_csv('rejected_images.csv', index=False)

print(f"Done! Kept {len(best_images)} best images.")
print(f"Moved {len(rejected_images)} duplicate artifact-heavy images to '{bad_folder}/'.")