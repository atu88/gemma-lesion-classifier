import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Parameters
image_folder = "images"
metadata_path = "ham_bcn_metadata.csv"
train_folder = "train"
test_folder = "test"
output_train_metadata_path = "train_metadata.csv"
output_test_metadata_path = "test_metadata.csv"

# Create folders for images
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)

# Fill NaN values with a label "Unknown"
df['diagnosis_3'] = df['diagnosis_3'].fillna("Unknown")

# Stratify the lesions
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['diagnosis_3'],
    random_state=42
)

# Copy images to respective folders
def copy_images(sub_df, destination_folder):
    for isic_id in sub_df['isic_id']:
        image_name = f"{isic_id}.jpg" 
        src_path = os.path.join(image_folder, image_name)
        dst_path = os.path.join(destination_folder, image_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: {image_name} not found")

copy_images(train_df, train_folder)
copy_images(test_df, test_folder)

# Update fields on metadata to match training script
train_df = train_df.rename(columns={'isic_id': 'image', 'diagnosis_3': 'label'})
train_df.to_csv(output_train_metadata_path, index=False)
test_df = test_df.rename(columns={'isic_id': 'image', 'diagnosis_3': 'label'})
test_df.to_csv(output_test_metadata_path, index=False)

print("Split complete.")
print(f"Train images: {len(train_df)} | Test images: {len(test_df)}")
print(f"Train metadata saved to: {output_train_metadata_path}")
print(f"Test metadata saved to: {output_test_metadata_path}")