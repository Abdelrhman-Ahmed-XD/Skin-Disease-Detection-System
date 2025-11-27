# Main_Pipeline.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# ============================================================
# Import Techniques
# ============================================================
sys.path.append(os.path.join(os.path.dirname(__file__), "Components", "Preprocessing_Techniques"))
from Hair_And_Artifact_Removal_Algorithm import remove_hair_and_artifacts
from Denoising_Algorithm import denoise_skin_image
from Color_Enhancement_Algorithm import apply_clahe_enhancement
from Resizing_Normalization_Algorithm import apply_resize_normalization

# ============================================================
# Directories
# ============================================================
base = os.path.dirname(__file__)

orig_dir = os.path.join(base, "Test Images", "Original_Images")
hair_removed_dir = os.path.join(base, "Test Results", "Hair & Artifact_Removal_Images")
denoised_dir = os.path.join(base, "Test Results", "Denoising_Images")
enhanced_dir = os.path.join(base, "Test Results", "Color_Enhancement_Images")
resized_norm_dir = os.path.join(base, "Test Results", "Resized & Normalized_Images")

plot_hair = os.path.join(base, "Test Results", "Before_VS_After_Hair_Artifact_Removal")
plot_denoise = os.path.join(base, "Test Results", "Before_VS_After_Denoising")
plot_enhance = os.path.join(base, "Test Results", "Before VS After_Color_Enhancement")
plot_resize_norm = os.path.join(base, "Test Results", "Before_VS_After_Resize_Normalization")
plot_final = os.path.join(base, "Test Results", "Original_VS_Final_Result_Images")

for folder in [hair_removed_dir, denoised_dir, enhanced_dir, resized_norm_dir,
               plot_hair, plot_denoise, plot_enhance, plot_resize_norm, plot_final]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# COMPARISON IMAGE CREATOR
# ============================================================
def create_comparison(original, processed, technique_name):
    H = 500

    def to_rgb(img):
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    original = to_rgb(original)
    processed = to_rgb(processed)

    def resize_and_pad(img):
        h, w = img.shape[:2]
        if h == 0: h = 1
        scale = H / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, H))
        return resized, new_w

    orig_resized, w1 = resize_and_pad(original)
    proc_resized, w2 = resize_and_pad(processed)
    max_w = max(w1, w2)
    if w1 < max_w:
        orig_resized = cv2.copyMakeBorder(orig_resized, 0, 0, 0, max_w - w1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if w2 < max_w:
        proc_resized = cv2.copyMakeBorder(proc_resized, 0, 0, 0, max_w - w2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    title_h = 100
    w = max_w * 2 + 80
    canvas = np.ones((H + title_h + 100, w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 3
    text_size = cv2.getTextSize(technique_name, font, font_scale, thickness)[0]
    while text_size[0] > w - 40 and font_scale > 0.7:
        font_scale -= 0.05
        text_size = cv2.getTextSize(technique_name, font, font_scale, thickness)[0]

    x = (w - text_size[0]) // 2
    cv2.putText(canvas, technique_name, (x, 70), font, font_scale, (0, 0, 0), thickness)
    canvas[title_h:title_h + H, 40:40 + max_w] = orig_resized
    canvas[title_h:title_h + H, 40 + max_w + 40:40 + max_w + 40 + max_w] = proc_resized
    cv2.putText(canvas, "Original", (max_w // 2 - 70, title_h + H + 60), font, 1.3, (0, 0, 0), 3)
    cv2.putText(canvas, "Result", (max_w + max_w // 2 + 30, title_h + H + 60), font, 1.3, (0, 120, 0), 3)

    return canvas

# ============================================================
# MAIN LOOP
# ============================================================
image_paths = set()
for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG"]:
    image_paths.update(Path(orig_dir).rglob(ext))
images = sorted(list(image_paths))

print(f"Found {len(images)} images. Starting processing...\n")

for img_path in tqdm(images, desc="Processing Images", unit="img", colour="green"):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    name = img_path.stem

    # Individual techniques (each applied directly on original)
    no_hair, _ = remove_hair_and_artifacts(original.copy())
    denoised = denoise_skin_image(original.copy())
    enhanced = apply_clahe_enhancement(original.copy())
    resized_norm = apply_resize_normalization(original.copy())

    # Full sequential pipeline
    step1 = no_hair
    step2 = denoise_skin_image(step1.copy())
    step3 = apply_clahe_enhancement(step2.copy())
    full = apply_resize_normalization(step3.copy())

    # Save processed images
    cv2.imwrite(os.path.join(hair_removed_dir, f"{name}.jpg"), cv2.cvtColor(no_hair, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(denoised_dir, f"{name}.jpg"), cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(enhanced_dir, f"{name}.jpg"), cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(resized_norm_dir, f"{name}.jpg"), cv2.cvtColor(resized_norm, cv2.COLOR_RGB2BGR))

    # Save comparison images with professional titles
    cv2.imwrite(os.path.join(plot_hair, f"{name}_comparison.jpg"),
                cv2.cvtColor(create_comparison(original, no_hair, "Hair & Artifact Removal (DullRazor + Inpainting)"),
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(plot_denoise, f"{name}_comparison.jpg"),
                cv2.cvtColor(create_comparison(original, denoised, "Denoising (Bilateral + Wavelet)"),
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(plot_enhance, f"{name}_comparison.jpg"),
                cv2.cvtColor(create_comparison(original, enhanced, "Color Enhancement (CLAHE-HSV)"),
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(plot_resize_norm, f"{name}_comparison.jpg"),
                cv2.cvtColor(create_comparison(original, resized_norm, "Resize 224x224 + Normalization"),
                             cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(plot_final, f"{name}_full_comparison.jpg"),
                cv2.cvtColor(create_comparison(original, full,
                                               "Full Pipeline ( ( DullRazor + Inpainting ) -> ( Bilateral + Wavelet ) -> CLAHE-HSV -> ( Resize 224x224 + Normalization ) )"),
                             cv2.COLOR_RGB2BGR))

print("\nAll 4 techniques + full pipeline completed successfully!")
print("Check the 'Test Results' folder — you now have perfect before/after comparisons for every step.")