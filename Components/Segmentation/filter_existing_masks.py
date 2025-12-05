# filter_existing_masks.py
"""
Filter existing masks to keep only those belonging to the 4 selected classes
MUCH faster than regenerating - takes only 2-3 minutes!
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("Filter Existing Masks for 4 Selected Classes")
print("=" * 80)

# ============================================================
# Configuration
# ============================================================
# Where your original masks are stored
ORIGINAL_MASKS_DIR = "../../Dataset/ISIC2019/masks_sam_generated"

# Where your 4-class images are
SELECTED_CLASSES = ['BCC', 'MEL', 'BKL', 'NV']
CLASS_FOLDERS = [f"Dataset/ISIC2019/{cls}" for cls in SELECTED_CLASSES]

# Output directory for filtered masks
OUTPUT_MASKS_DIR = "../../Dataset/ISIC2019/masks_4classes_filtered"

# ============================================================
# Verify Setup
# ============================================================
print("\n🔍 Verifying setup...")

# Check if original masks exist
if not os.path.exists(ORIGINAL_MASKS_DIR):
    print(f"\n❌ ERROR: Original masks directory not found!")
    print(f"   Expected: {ORIGINAL_MASKS_DIR}")
    print("\n   Where are your existing masks stored?")
    print("   Please update ORIGINAL_MASKS_DIR in this script.")
    exit(1)

original_mask_count = len(list(Path(ORIGINAL_MASKS_DIR).glob("*.png")))
print(f"   ✓ Found {original_mask_count} masks in {ORIGINAL_MASKS_DIR}")

# Check disease folders
missing_folders = []
for folder in CLASS_FOLDERS:
    if not os.path.exists(folder):
        missing_folders.append(folder)

if missing_folders:
    print(f"\n❌ ERROR: Missing disease folders:")
    for folder in missing_folders:
        print(f"   - {folder}")
    exit(1)

print(f"   ✓ All 4 disease folders found")

# Create output directory
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

# ============================================================
# Collect Image Names from 4 Classes
# ============================================================
print("\n📂 Collecting image names from 4 classes...")

selected_image_names = set()
class_counts = {}

for disease in SELECTED_CLASSES:
    disease_path = f"Dataset/ISIC2019/{disease}"

    # Get all images in this class
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(disease_path).glob(ext))

    # Extract just the base names (without extension)
    for img_path in image_files:
        selected_image_names.add(img_path.stem)

    class_counts[disease] = len(image_files)
    print(f"   {disease}: {len(image_files)} images")

print(f"\n   Total unique images in 4 classes: {len(selected_image_names)}")

# ============================================================
# Filter and Copy Masks
# ============================================================
print("\n📋 Filtering masks...")

copied = 0
missing = 0
skipped = 0

for img_name in tqdm(selected_image_names, desc="Copying masks", colour="green"):
    # Expected mask filename
    mask_name = f"{img_name}_segmentation.png"

    # Source mask path
    src_mask = os.path.join(ORIGINAL_MASKS_DIR, mask_name)

    # Destination mask path
    dst_mask = os.path.join(OUTPUT_MASKS_DIR, mask_name)

    # Check if mask exists in original location
    if not os.path.exists(src_mask):
        missing += 1
        continue

    # Check if already copied
    if os.path.exists(dst_mask):
        skipped += 1
        continue

    # Copy mask
    try:
        shutil.copy2(src_mask, dst_mask)
        copied += 1
    except Exception as e:
        print(f"\n⚠ Error copying {mask_name}: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("✓ Filtering Complete!")
print("=" * 80)

print(f"\n📊 Results:")
print(f"   Images in 4 classes:  {len(selected_image_names)}")
print(f"   ✓ Masks copied:       {copied}")
print(f"   ⊙ Already existed:    {skipped}")
print(f"   ✗ Missing masks:      {missing}")

if missing > 0:
    print(f"\n⚠ {missing} images don't have corresponding masks")
    print("   These images will be skipped in training")

print(f"\n📁 Filtered masks saved to:")
print(f"   {OUTPUT_MASKS_DIR}")

print("\n" + "=" * 80)
print("✓ Next Steps:")
print("   1. Update split_4classes.py to use the filtered masks")
print(f"      Change MASKS_DIR to: {OUTPUT_MASKS_DIR}")
print("   2. Run: python split_4classes.py")
print("=" * 80)

# ============================================================
# Optional: Show missing images
# ============================================================
if missing > 0 and missing <= 20:
    print(f"\n📄 Images without masks:")
    count = 0
    for img_name in selected_image_names:
        mask_name = f"{img_name}_segmentation.png"
        src_mask = os.path.join(ORIGINAL_MASKS_DIR, mask_name)
        if not os.path.exists(src_mask):
            print(f"   - {img_name}")
            count += 1
            if count >= 20:
                print(f"   ... and {missing - 20} more")
                break