# Components/Preprocessing_Techniques/Hair_And_Artifact_Removal_Algorithm.py
import cv2
import numpy as np

def remove_hair_and_artifacts(image):
    """
    Removes hair and artifacts using a modified DullRazor approach followed by inpainting.
    - Detects and masks hair while preserving lesion areas.
    - Applies morphological operations for mask refinement.
    - Uses inpainting to fill masked regions naturally.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 1: Apply black-hat transform with a small kernel to detect hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Step 2: Threshold to create hair mask, avoiding dark lesions
    thresh_val = max(20, blackhat.mean() + blackhat.std() * 0.8)
    _, hair_mask = cv2.threshold(blackhat, thresh_val, 255, cv2.THRESH_BINARY)

    # Step 3: Remove large connected components to protect lesions
    num_labels, labels = cv2.connectedComponents(hair_mask)
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area > 800:
            hair_mask[labels == i] = 0

    # Step 4: Close gaps and dilate the mask
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    hair_mask = cv2.dilate(hair_mask,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)), iterations=1)

    # Step 5: Remove small specks
    num_labels, labels = cv2.connectedComponents(hair_mask)
    for i in range(1, num_labels):
        if np.sum(labels == i) < 60:
            hair_mask[labels == i] = 0

    # Step 6: Inpaint masked areas
    result = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result, hair_mask