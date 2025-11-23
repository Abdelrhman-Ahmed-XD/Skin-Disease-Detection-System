# Components/Preprocessing_Techniques/Denoising_Algorithm.py
import cv2
import numpy as np

def denoise_skin_image(image):
    """
    Applies denoising techniques for skin images.
    - Uses bilateral filtering to preserve edges while reducing noise.
    - Applies wavelet transform for further noise removal in flat areas.
    - Enhances sharpness with unsharp masking.
    """
    # Step 1: Apply bilateral filter
    bil = cv2.bilateralFilter(image, d=7, sigmaColor=10, sigmaSpace=10)

    # Step 2: Wavelet denoising
    img_float = bil.astype(np.float32)

    import pywt
    coeffs = pywt.dwt2(img_float, 'db1')
    cA, (cH, cV, cD) = coeffs

    threshold = 12
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold * 1.5, mode='soft')

    coeffs_clean = cA, (cH, cV, cD)
    denoised_float = pywt.idwt2(coeffs_clean, 'db1')

    denoised = np.clip(denoised_float, 0, 255).astype(np.uint8)

    # Step 3: Apply unsharp mask for sharpening
    gaussian = cv2.GaussianBlur(denoised, (0,0), 1.0)
    final = cv2.addWeighted(denoised, 1.6, gaussian, -0.6, 0)

    return final