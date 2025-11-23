# Components/Preprocessing_Techniques/Resizing_Normalization_Algorithm.py
import cv2
import numpy as np

def apply_resize_normalization(image):
    """
    Applies resizing and normalization as the final preprocessing step.
    - Resizes the image to 224x224 pixels, suitable for models like ResNet and EfficientNet.
    - Normalizes using mean and standard deviation values from the ISIC-2019 dataset.
    - Returns the image in uint8 format after denormalization for saving and visual comparison.
    """
    # Step 1: Resize to 224x224
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # Step 2: Normalize using ISIC-2019 statistics
    img_float = resized.astype(np.float32) / 255.0
    mean = np.array([0.7040, 0.5475, 0.5403])  # R, G, B
    std  = np.array([0.2967, 0.2923, 0.3039])   # R, G, B
    normalized = (img_float - mean) / std

    # Step 3: Denormalize to uint8 for saving and comparison
    denormalized = np.clip(normalized * std + mean, 0, 1)
    result = (denormalized * 255).astype(np.uint8)

    return result