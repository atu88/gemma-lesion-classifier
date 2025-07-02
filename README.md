Image Quality Filter for BCN20000 Dataset
This script processes images from the BCN20000 skin lesion dataset to identify and retain the best-quality image per lesion, based on blur, entropy, and edge density scores. Duplicate or artifact-heavy images (e.g., with dark borders) are moved to a separate folder.

Features
Scores images using:

Blur (Laplacian variance)

Shannon entropy

Edge density (Canny edge detection)

Detects and deprioritizes images with dark borders.

Keeps the best image per lesion and moves lower-quality duplicates to a duplicates folder.

Outputs cleaned metadata and list of rejected images.

Requirements
Python 3.x

Packages: opencv-python, numpy, pandas, scikit-image

Install dependencies via pip:

bash
Copy
Edit
pip install opencv-python numpy pandas scikit-image
Usage
Place your images in the folder specified by image_folder (default: images/).

Provide the metadata CSV file path in metadata_file (default: bcn20000_metadata.csv).

Run the script:

bash
Copy
Edit
python your_script.py
After running:

Best images remain in the image folder.

Duplicate or low-quality images are moved to the duplicates/ folder.

New metadata file clean_metadata_bcn.csv with best images is saved.

rejected_images.csv lists moved image IDs.

Parameters
Adjust scoring weights (w_blur, w_entropy, w_edges) to tweak the importance of each metric.

Modify threshold and dark_pixel_ratio in has_dark_border() to customize dark border detection sensitivity.
