# SAM_Mask_Generator/generate_masks_isic2019.py
"""
Generate segmentation masks for ISIC-2019 using SAM + YOLO
Optimized for Windows + PyCharm
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO

print("=" * 70)
print("ISIC-2019 Automatic Mask Generation with SAM + YOLO")
print("=" * 70)


# ============================================================
# Configuration
# ============================================================
class Config:
    # Paths (adjust if needed)
    ISIC_IMAGES_DIR = "../Dataset/ISIC2019/all_images"
    OUTPUT_MASKS_DIR = "../Dataset/ISIC2019/masks_sam_generated"

    # SAM Model
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Processing
    SKIP_EXISTING = True  # Resume if interrupted
    BATCH_LOG_FREQUENCY = 100  # Print progress every N images


# ============================================================
# Initialize Models
# ============================================================
print(f"\nDevice: {Config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

print("\nLoading models...")

# Load SAM
if not os.path.exists(Config.SAM_CHECKPOINT):
    print(f"\n❌ ERROR: SAM weights not found!")
    print(f"   Expected: {Config.SAM_CHECKPOINT}")
    print(f"   Please download from: https://github.com/facebookresearch/segment-anything")
    sys.exit(1)

sam = sam_model_registry[Config.SAM_MODEL_TYPE](checkpoint=Config.SAM_CHECKPOINT)
sam.to(device=Config.DEVICE)
sam_predictor = SamPredictor(sam)

# Load YOLO
print("Loading YOLO...")
yolo_model = YOLO('yolov8m.pt')  # Auto-downloads if not exists
yolo_model.to(Config.DEVICE)

print("✓ Models loaded successfully\n")

# Create output directory
os.makedirs(Config.OUTPUT_MASKS_DIR, exist_ok=True)


# ============================================================
# Mask Generation Function
# ============================================================
def generate_mask_for_image(image_path):
    """Generate segmentation mask using YOLO + SAM"""

    try:
        # Read image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        # Method: YOLO detection + SAM segmentation
        results = yolo_model(image_rgb, verbose=False)[0]

        if len(results.boxes) > 0:
            # Get largest detection (likely the lesion)
            boxes = results.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            bbox = boxes[largest_idx]

            # Use bbox as SAM prompt
            sam_predictor.set_image(image_rgb)
            masks, scores, _ = sam_predictor.predict(
                box=bbox[None, :],
                multimask_output=False
            )
            mask = masks[0]

        else:
            # Fallback: center point (dermoscopy images are usually centered)
            center_point = np.array([[width // 2, height // 2]])

            sam_predictor.set_image(image_rgb)
            masks, scores, _ = sam_predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                multimask_output=True
            )
            mask = masks[np.argmax(scores)]

        # Convert to binary mask
        mask = mask.astype(np.uint8) * 255

        # Optional: morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    except Exception as e:
        print(f"\n⚠ Error: {image_path.name}: {str(e)}")
        return None


# ============================================================
# Main Processing Loop
# ============================================================
# Get all images
image_paths = list(Path(Config.ISIC_IMAGES_DIR).glob("*.jpg"))
image_paths += list(Path(Config.ISIC_IMAGES_DIR).glob("*.jpeg"))
image_paths += list(Path(Config.ISIC_IMAGES_DIR).glob("*.png"))

print(f"Found {len(image_paths)} images to process\n")

if len(image_paths) == 0:
    print(f"❌ ERROR: No images found in {Config.ISIC_IMAGES_DIR}")
    print("   Did you run collect_all_images.py first?")
    sys.exit(1)

# Process each image
successful = 0
skipped = 0
failed = 0
start_time = None

for idx, img_path in enumerate(tqdm(image_paths, desc="Generating Masks", unit="img", colour="blue")):
    # Track time
    if idx == 0:
        import time

        start_time = time.time()

    # Output mask path
    mask_name = img_path.stem + "_segmentation.png"
    mask_path = os.path.join(Config.OUTPUT_MASKS_DIR, mask_name)

    # Skip if exists
    if Config.SKIP_EXISTING and os.path.exists(mask_path):
        skipped += 1
        continue

    # Generate mask
    mask = generate_mask_for_image(img_path)

    if mask is not None:
        cv2.imwrite(mask_path, mask)
        successful += 1
    else:
        failed += 1

    # Periodic progress update
    if (idx + 1) % Config.BATCH_LOG_FREQUENCY == 0 and start_time:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed
        remaining = (len(image_paths) - idx - 1) / rate if rate > 0 else 0
        print(f"\n  Progress: {idx + 1}/{len(image_paths)} | "
              f"Rate: {rate:.1f} img/s | "
              f"ETA: {remaining / 3600:.1f}h")

# Final summary
print("\n" + "=" * 70)
print("Mask Generation Complete!")
print("=" * 70)
print(f"✓ Successfully generated: {successful}")
print(f"⊙ Skipped (already exists): {skipped}")
print(f"✗ Failed: {failed}")
print(f"\nMasks saved to: {Config.OUTPUT_MASKS_DIR}")

# Show some examples
if successful > 0:
    print("\nSample mask files:")
    for mask_file in list(Path(Config.OUTPUT_MASKS_DIR).glob("*.png"))[:5]:
        print(f"  - {mask_file.name}")

print("=" * 70)