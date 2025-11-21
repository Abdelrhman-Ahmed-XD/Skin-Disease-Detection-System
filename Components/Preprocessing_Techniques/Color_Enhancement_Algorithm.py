# Components/Preprocessing_Techniques/Color_Enhancement_Algorithm.py
import cv2

def apply_clahe_enhancement(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v_enhanced = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    return result