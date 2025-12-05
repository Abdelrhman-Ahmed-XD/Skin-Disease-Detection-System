# generate_masks_4classes.py
"""
Step 4: Generate segmentation masks for 4 disease classes using SAM + YOLO
Optimized for Windows + PyCharm + CUDA
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

print("=" * 80)
print("Step 4: Automatic Mask Generation with SAM + YOLO")
print("=" * 80)


# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths
    IMAGES_DIR = "Dataset/ISIC2019/selected_4classes/images"
    OUTPUT_MASKS_DIR = "../../Dataset/ISIC2019/masks_sam_generated"
    SAM_CHECKPOINT = "SAM_Mask_Generator/sam_vit_h_4b8939.pth"

    # Model
    SAM_MODEL_TYPE = "vit_h"

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Processing
    SKIP_EXISTING = True
    LOG_FREQUENCY = 100


# ============================================================
# Verify Setup
# ============================================================
print(f"\n🔍 Verifying setup...")

# Check images directory
if not os.path.exists(Config.IMAGES_DIR):
    print(f"\n❌ ERROR: Images directory not found!")
    print(f"   Expected: {Config.IMAGES_DIR}")
    print(f"   Please run collect_4classes.py first (Step 2)")
    sys.exit(1)

image_count = len(list(Path(Config.IMAGES_DIR).glob("*.jpg"))) + \
              len(list(Path(Config.IMAGES_DIR).glob("*.png")))

if image_count == 0:
    print(f"\n❌ ERROR: No images found in {Config.IMAGES_DIR}")
    print(f"   Please run collect_4classes.py first (Step 2)")
    sys.exit(1)

print(f"   ✓ Found {image_count} images")

# Check SAM weights
if not os.path.exists(Config.SAM_CHECKPOINT):
    print(f"\n❌ ERROR: SAM weights not found!")
    print(f"   Expected: {Config.SAM_CHECKPOINT}")
    print(f"   Please run download_sam_weights.py first (Step 3)")
    sys.exit(1)

sam_size = os.path.getsize(Config.SAM_CHECKPOINT) / 1024 ** 3
print(f"   ✓ SAM weights found ({sam_size:.2f} GB)")

# Create output directory
os.makedirs(Config.OUTPUT_MASKS_DIR, exist_ok=True)

# ============================================================
# Initialize Models
# ============================================================
print(f"\n🚀 Initializing models...")
print(f"   Device: {Config.DEVICE}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("   ⚠ Running on CPU (this will be VERY slow)")
    print("   Estimated time: 10-15 hours")

print("\n   Loading SAM model...")
try:
    sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
    sam.to(device=Config.DEVICE)
    sam_predictor = SamPredictor(sam)
    print("   ✓ SAM loaded successfully")
except Exception as e:
    print(f"\n❌ ERROR loading SAM: {e}")
    sys.exit(1)

print("   Loading YOLO model...")
try:
    yolo_model = YOLO('../../yolov8m.pt')  # Auto-downloads if needed
    print("   ✓ YOLO loaded successfully")
except Exception as e:
    print(f"\n❌ ERROR loading YOLO: {e}")
    sys.exit(1)


# ============================================================
# Mask Generation Function
# ============================================================
def generate_mask(image_path):
    """Generate segmentation mask using YOLO + SAM"""
    try:
        # Read image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # Step 1: Detect lesion with YOLO
        results = yolo_model(image_rgb, verbose=False)[0]

        if len(results.boxes) > 0:
            # Get largest detection
            boxes = results.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            bbox = boxes[largest_idx]

            # Step 2: Use bbox as SAM prompt
            sam_predictor.set_image(image_rgb)
            masks, scores, _ = sam_predictor.predict(
                box=bbox[None, :],
                multimask_output=False
            )
            mask = masks[0]

        else:
            # Fallback: center point (dermoscopy images are centered)
            center_point = np.array([[width // 2, height // 2]])

            sam_predictor.set_image(image_rgb)
            masks, scores, _ = sam_predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                multimask_output=True
            )
            mask = masks[np.argmax(scores)]

        # Convert to binary mask (0 or 255)
        mask = mask.astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    except Exception as e:
        return None


# ============================================================
# Process All Images
# ============================================================
print("\n" + "=" * 80)
print("🎨 Generating Masks")
print("=" * 80)

# Get all images
image_paths = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    image_paths.extend(Path(Config.IMAGES_DIR).glob(ext))

print(f"\nTotal images to process: {len(image_paths)}")

# Check for existing masks
existing_masks = len(list(Path(Config.OUTPUT_MASKS_DIR).glob("*.png")))
if existing_masks > 0 and Config.SKIP_EXISTING:
    print(f"Existing masks found: {existing_masks}")
    print("Will skip images that already have masks\n")

# Statistics
successful = 0
skipped = 0
failed = 0
start_time = time.time()

# Process each image
print("Processing...")
for idx, img_path in enumerate(tqdm(image_paths, desc="Generating masks", unit="img", colour="blue")):
    # Output mask path
    mask_name = f"{img_path.stem}_segmentation.png"
    mask_path = os.path.join(Config.OUTPUT_MASKS_DIR, mask_name)

    # Skip if exists
    if Config.SKIP_EXISTING and os.path.exists(mask_path):
        skipped += 1
        continue

    # Generate mask
    mask = generate_mask(img_path)

    if mask is not None:
        cv2.imwrite(mask_path, mask)
        successful += 1
    else:
        failed += 1

    # Periodic update
    if (idx + 1) % Config.LOG_FREQUENCY == 0:
        elapsed = time.time() - start_time
        rate = (successful + skipped + failed) / elapsed
        remaining = (len(image_paths) - idx - 1) / rate if rate > 0 else 0

        print(f"\n  Progress: {idx + 1}/{len(image_paths)} | "
              f"Rate: {rate:.1f} img/s | "
              f"ETA: {remaining / 3600:.1f}h | "
              f"Success: {successful} | "
              f"Skipped: {skipped} | "
              f"Failed: {failed}")

# Calculate total time
total_time = time.time() - start_time

# Final summary
print("\n" + "=" * 80)
print("✓ Mask Generation Complete!")
print("=" * 80)

print(f"\n📊 Results:")
print(f"   Total images:      {len(image_paths)}")
print(f"   ✓ Generated:       {successful}")
print(f"   ⊙ Skipped:         {skipped} (already existed)")
print(f"   ✗ Failed:          {failed}")
print(f"\n⏱ Time:")
print(f"   Total time:        {total_time / 3600:.2f} hours")
print(f"   Average speed:     {(successful + skipped) / total_time:.1f} images/second")

print(f"\n📁 Masks saved to:")
print(f"   {Config.OUTPUT_MASKS_DIR}")

# Show examples
print(f"\n📄 Sample mask files:")
for mask_file in list(Path(Config.OUTPUT_MASKS_DIR).glob("*.png"))[:5]:
    print(f"   - {mask_file.name}")

print("\n" + "=" * 80)
print("✓ Ready for Step 5: Dataset Splitting")
print("=" * 80)