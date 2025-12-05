# split_4classes.py
"""
Step 5: Split 4-class dataset into train/val with stratification
Maintains class balance in both sets
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("=" * 80)
print("Step 5: Splitting Dataset (Stratified by Disease Class)")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
IMAGES_DIR = "Dataset/ISIC2019/all_4classes"
MASKS_DIR = "Dataset/ISIC2019/masks_4classes_filtered"  # Using filtered masks!
TRAIN_IMG_DIR = "Dataset/ISIC2019/train/images"
TRAIN_MASK_DIR = "Dataset/ISIC2019/train/masks"
VAL_IMG_DIR = "Dataset/ISIC2019/val/images"
VAL_MASK_DIR = "Dataset/ISIC2019/val/masks"

TRAIN_RATIO = 0.9  # 90% train, 10% validation
RANDOM_SEED = 42

disease_classes = ['BCC', 'MEL', 'BKL', 'NV']

# ============================================================
# Verify Setup
# ============================================================
print(f"\n🔍 Verifying setup...")

if not os.path.exists(IMAGES_DIR):
    print(f"\n❌ ERROR: Images directory not found: {IMAGES_DIR}")
    print("   Please run collect_4classes.py first (Step 2)")
    exit(1)

if not os.path.exists(MASKS_DIR):
    print(f"\n❌ ERROR: Masks directory not found: {MASKS_DIR}")
    print("   Please run generate_masks_4classes.py first (Step 4)")
    exit(1)

print("   ✓ Directories verified")

# ============================================================
# Create Output Directories
# ============================================================
for directory in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# Find Valid Image-Mask Pairs
# ============================================================
print(f"\n📂 Finding valid image-mask pairs...")

image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
    image_paths.extend(Path(IMAGES_DIR).glob(ext))

print(f"   Total images: {len(image_paths)}")

# Match images with masks and determine disease class
valid_pairs = []
missing_masks = []

for img_path in tqdm(image_paths, desc="Matching pairs", colour="blue"):
    # Expected mask name
    mask_name = f"{img_path.stem}_segmentation.png"
    mask_path = Path(MASKS_DIR) / mask_name

    if not mask_path.exists():
        missing_masks.append(img_path.name)
        continue

    # Determine disease class from original folder
    disease = None
    for d in disease_classes:
        original_path = Path("Dataset/ISIC2019") / d / img_path.name
        if original_path.exists():
            disease = d
            break

    if disease is None:
        disease = "UNKNOWN"

    valid_pairs.append({
        'image': img_path.name,
        'mask': mask_name,
        'disease': disease
    })

# Create DataFrame
df = pd.DataFrame(valid_pairs)

print(f"\n   ✓ Valid pairs: {len(df)}")
if missing_masks:
    print(f"   ⚠ Missing masks: {len(missing_masks)}")
    if len(missing_masks) <= 5:
        for img in missing_masks:
            print(f"      - {img}")

if len(df) == 0:
    print("\n❌ ERROR: No valid image-mask pairs found!")
    print("   Make sure masks are named: [image_name]_segmentation.png")
    exit(1)

# ============================================================
# Display Distribution
# ============================================================
print(f"\n📊 Dataset Distribution:")
print("-" * 50)

for disease in disease_classes:
    count = (df['disease'] == disease).sum()
    pct = (count / len(df)) * 100
    print(f"   {disease:5s}: {count:6d} images ({pct:5.1f}%)")

print("-" * 50)
print(f"   TOTAL: {len(df):6d} images (100.0%)")

# ============================================================
# Stratified Split
# ============================================================
print(f"\n✂️ Splitting dataset ({TRAIN_RATIO * 100:.0f}% train, {(1 - TRAIN_RATIO) * 100:.0f}% val)...")

train_df, val_df = train_test_split(
    df,
    train_size=TRAIN_RATIO,
    random_state=RANDOM_SEED,
    stratify=df['disease'],
    shuffle=True
)

print(f"   Train set: {len(train_df)} images")
print(f"   Val set:   {len(val_df)} images")

# Show distribution in each set
print(f"\n📊 Training Set Distribution:")
for disease in disease_classes:
    count = (train_df['disease'] == disease).sum()
    pct = (count / len(train_df)) * 100
    print(f"   {disease:5s}: {count:6d} ({pct:5.1f}%)")

print(f"\n📊 Validation Set Distribution:")
for disease in disease_classes:
    count = (val_df['disease'] == disease).sum()
    pct = (count / len(val_df)) * 100
    print(f"   {disease:5s}: {count:6d} ({pct:5.1f}%)")

# ============================================================
# Copy Files
# ============================================================
print(f"\n📋 Copying files...")

# Training set
print("   Copying training set...")
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="   Train", colour="green", leave=False):
    # Copy image
    shutil.copy2(
        os.path.join(IMAGES_DIR, row['image']),
        os.path.join(TRAIN_IMG_DIR, row['image'])
    )
    # Copy mask
    shutil.copy2(
        os.path.join(MASKS_DIR, row['mask']),
        os.path.join(TRAIN_MASK_DIR, row['mask'])
    )

print(f"   ✓ Copied {len(train_df)} training pairs")

# Validation set
print("   Copying validation set...")
for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="   Val", colour="cyan", leave=False):
    # Copy image
    shutil.copy2(
        os.path.join(IMAGES_DIR, row['image']),
        os.path.join(VAL_IMG_DIR, row['image'])
    )
    # Copy mask
    shutil.copy2(
        os.path.join(MASKS_DIR, row['mask']),
        os.path.join(VAL_MASK_DIR, row['mask'])
    )

print(f"   ✓ Copied {len(val_df)} validation pairs")

# ============================================================
# Verify Split
# ============================================================
print(f"\n🔍 Verifying split...")

train_imgs = len(list(Path(TRAIN_IMG_DIR).glob("*.*")))
train_masks = len(list(Path(TRAIN_MASK_DIR).glob("*.*")))
val_imgs = len(list(Path(VAL_IMG_DIR).glob("*.*")))
val_masks = len(list(Path(VAL_MASK_DIR).glob("*.*")))

print(f"\n   Training:")
print(f"      Images: {train_imgs}")
print(f"      Masks:  {train_masks}")
print(f"      Match:  {'✓' if train_imgs == train_masks else '✗ MISMATCH!'}")

print(f"\n   Validation:")
print(f"      Images: {val_imgs}")
print(f"      Masks:  {val_masks}")
print(f"      Match:  {'✓' if val_imgs == val_masks else '✗ MISMATCH!'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✓ Dataset Split Complete!")
print("=" * 80)

print(f"\n📊 Final Statistics:")
print(f"   Total pairs:    {len(df)}")
print(f"   Training:       {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
print(f"   Validation:     {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")

print(f"\n📁 Output Directories:")
print(f"   Train images:   {TRAIN_IMG_DIR}")
print(f"   Train masks:    {TRAIN_MASK_DIR}")
print(f"   Val images:     {VAL_IMG_DIR}")
print(f"   Val masks:      {VAL_MASK_DIR}")

print("\n" + "=" * 80)
print("✓ Ready for Step 6: Training")
print("=" * 80)