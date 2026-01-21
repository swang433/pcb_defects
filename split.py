import shutil
from pathlib import Path
import random

# Paths
images_dir = Path("PCB_DATASET\images")
labels_dir = Path("PCB_DATASET\labels")
output_dir = Path("PCB_DATASET\yolo_dataset")

# Create structure
for split in ['train', 'val']:
    (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

# Get all images from subdirectories
all_images = []
for img_file in images_dir.rglob("*.jpg"):
    # Check if corresponding label exists
    label_file = labels_dir / (img_file.stem + ".txt")
    if label_file.exists():
        all_images.append((img_file, label_file))

print(f"Found {len(all_images)} image-label pairs")

# Shuffle and split 80/20
random.seed(42)
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)

train_data = all_images[:split_idx]
val_data = all_images[split_idx:]

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Copy files
for img_path, label_path in train_data:
    shutil.copy(img_path, output_dir / 'train' / 'images' / img_path.name)
    shutil.copy(label_path, output_dir / 'train' / 'labels' / label_path.name)

for img_path, label_path in val_data:
    shutil.copy(img_path, output_dir / 'val' / 'images' / img_path.name)
    shutil.copy(label_path, output_dir / 'val' / 'labels' / label_path.name)

print("Dataset organized!")