# ==============================================================================
# SKINSIGHT BACKEND — COMPLETE app.py
# Includes: Auth, Email, OTP, Password Reset, Email Change + AI Predict Route
# ==============================================================================

import os
import io
import cv2
import torch
import joblib
import requests
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import cloudinary
import cloudinary.uploader

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from scipy.stats import entropy as scipy_entropy

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Firebase Admin ────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
import json

# ── Email ─────────────────────────────────────────────────────────────────────
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Disease info ──────────────────────────────────────────────────────────────
from disease_info import DISEASE_INFO
from email_templates import get_otp_email_html, get_password_reset_html, get_email_change_html

load_dotenv()

# ==============================================================================
# APP SETUP
# ==============================================================================

app = Flask(__name__)
CORS(app, origins="*")

# ==============================================================================
# FIREBASE INIT
# ==============================================================================

_firebase_initialized = False

def get_firebase():
    global _firebase_initialized
    if not _firebase_initialized:
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            cred_dict = json.loads(service_account_json)
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
    return firestore.client()

# ==============================================================================
# CLOUDINARY CONFIG  (uses your existing keys from .env)
# ==============================================================================

# ==============================================================================
# CLOUDINARY CONFIG (Unsigned Upload Strategy)
# ==============================================================================

# Force environment variables to strip any hidden Windows characters
c_name = os.getenv("EXPO_PUBLIC_CLOUDINARY_CLOUD_NAME", "").strip()
c_preset = os.getenv("EXPO_PUBLIC_CLOUDINARY_UPLOAD_PRESET", "skinsight_uploads").strip()

# Only the cloud_name is required for unsigned preset uploads!
cloudinary.config(
    cloud_name=c_name,
    secure=True,
)

# ==============================================================================
# AI MODEL CONFIG
# ==============================================================================

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
IMAGE_SIZE  = 384
CLASSES     = ["NV", "MEL", "BKL", "BCC"]
MIN_LESION_AREA = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SkinSight] Compute device: {device}")

# Shared pre-processing for the 4 classifiers
infer_transforms = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Pre-processing for U-Net (256×256)
unet_transforms = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Model holders (loaded once at startup) ────────────────────────────────────
_unet        = None
_model_conv  = None
_model_res   = None
_model_den   = None
_model_max   = None
_boss_model  = None
_models_loaded = False


def _build_convnext():
    m = models.convnext_base()
    m.classifier[-1] = nn.Linear(1024, 4)
    return m.to(device)


def _build_resnext():
    m = models.resnext50_32x4d()
    m.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(2048, 4))
    return m.to(device)


def _build_densenet():
    m = models.densenet121()
    m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(1024, 4))
    return m.to(device)


class _MaxViTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxvit = models.maxvit_t()
        self.maxvit.classifier[-1] = nn.Linear(512, 4)

    def forward(self, x):
        return self.maxvit(x)


def _build_maxvit():
    return _MaxViTWrapper().to(device)


def _load_model(build_fn, filename):
    m = build_fn()
    path = os.path.join(MODELS_DIR, filename)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


def load_ai_models():
    """
    Called once when Flask starts (or on first predict request).
    Loads all 5 model files from the /models directory.
    """
    global _unet, _model_conv, _model_res, _model_den, _model_max, _boss_model, _models_loaded

    if _models_loaded:
        return

    print("[SkinSight] Loading AI models — this takes ~30s on first run...")

    # U-Net++ segmentation gatekeeper
    unet_path = os.path.join(MODELS_DIR, "U-Net++_EfficientNet-B4.pth")
    _unet = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    ckpt = torch.load(unet_path, map_location=device)
    # Strip compiled-model prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    _unet.load_state_dict(state)
    _unet.eval()
    print("  [+] U-Net++ loaded")

    # 4 classification models
    _model_conv = _load_model(_build_convnext,  "BEST_ConvNeXt-Base.pth")
    print("  [+] ConvNeXt-Base loaded")
    _model_res  = _load_model(_build_resnext,   "BEST_ResNeXt50.pth")
    print("  [+] ResNeXt50 loaded")
    _model_den  = _load_model(_build_densenet,  "BEST_DenseNet121.pth")
    print("  [+] DenseNet121 loaded")
    _model_max  = _load_model(_build_maxvit,    "BEST_MaxViT-T.pth")
    print("  [+] MaxViT-T loaded")

    # Meta-learner (scikit-learn stacking classifier)
    _boss_model = joblib.load(os.path.join(MODELS_DIR, "Skinsight_Meta_Boss.pkl"))
    print("  [+] Meta-Boss loaded")

    _models_loaded = True
    print("[SkinSight] All models ready.")


# Load models at startup (non-blocking if they fail — handled in route)
try:
    load_ai_models()
except Exception as e:
    print(f"[SkinSight] WARNING: Model loading failed at startup: {e}")
    print("[SkinSight] Models will be loaded on first request.")


# ==============================================================================
# AI PIPELINE FUNCTIONS
# ==============================================================================

def _smart_unet_crop(img_rgb: np.ndarray, padding: int = 60, max_ratio: float = 0.85):
    """
    Runs U-Net segmentation and applies the min/max area gates.
    Returns (cropped_img, mask_full, bbox, status_string)
    """
    img_t = unet_transforms(image=img_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        mask_raw = torch.sigmoid(_unet(img_t)).squeeze().cpu().numpy()

    mask_full = cv2.resize(
        (mask_raw > 0.25).astype(np.uint8),
        (img_rgb.shape[1], img_rgb.shape[0]),
    )

    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask_full, None, "REJECT_NORMAL_SKIN"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total = img_rgb.shape[0] * img_rgb.shape[1]

    if area < MIN_LESION_AREA:
        return None, mask_full, None, "REJECT_NORMAL_SKIN"

    if area > (total * max_ratio):
        return None, mask_full, None, "REJECT_TOO_LARGE"

    x, y, w, h = cv2.boundingRect(largest)
    y1 = max(0, y - padding)
    y2 = min(img_rgb.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(img_rgb.shape[1], x + w + padding)

    cropped = img_rgb[y1:y2, x1:x2]
    bbox = (x1, y1, x2, y2)
    return cropped, mask_full, bbox, "LESION_FOUND"


def _upload_mask_to_cloudinary(mask: np.ndarray) -> str | None:
    """
    Converts the binary U-Net mask to a coloured overlay PNG and
    uploads it to Cloudinary using an UNSIGNED preset (bypassing signature errors).
    """
    try:
        # Create a transparent RGBA overlay
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Matches your UI's var(--accent) #00e5ff
        # OpenCV uses BGR format: [Blue, Green, Red, Alpha/Opacity]
        overlay[mask == 1] = [255, 229, 0, 200]

        # Encode as PNG in memory
        success, buf = cv2.imencode(".png", overlay)
        if not success:
            return None

        # 🚀 Clean unsigned upload without restricted parameters
        result = cloudinary.uploader.unsigned_upload(
            file=buf.tobytes(),
            upload_preset=c_preset,  # Uses your "skinsight_uploads" preset
            folder="skinsight_masks"  # Saves it neatly into its own folder
        )

        return result.get("secure_url")

    except Exception as e:
        print(f"[SkinSight] Mask upload failed: {e}")
        return None

def _run_pipeline(img_rgb: np.ndarray, photo_type: str = "phone"):
    """
    Full AI pipeline:
      1. U-Net gating
      2. 4-model ensemble
      3. Meta-boss prediction
      4. Confidence thresholding → status
    Returns a dict ready to be JSON-serialised.
    """
    max_ratio = 0.35 if photo_type == "phone" else 0.85

    # ── STAGE 1: U-Net gate ───────────────────────────────────────────────────
    safe_crop, mask_full, bbox, unet_status = _smart_unet_crop(
        img_rgb, padding=60, max_ratio=max_ratio
    )

    if unet_status == "REJECT_NORMAL_SKIN":
        return {
            "status":        "no_lesion",
            "disease":       None,
            "disease_code":  None,
            "confidence":    0.0,
            "entropy":       0.0,
            "segmented_url": None,
            "description":   None,
            "tips":          [],
            "precautions":   [],
            "sources":       [],
            "message":       "No skin lesion detected. Please take a clearer, close-up photo of the affected area with good lighting.",
        }

    if unet_status == "REJECT_TOO_LARGE":
        return {
            "status":        "bad_photo",
            "disease":       None,
            "disease_code":  None,
            "confidence":    0.0,
            "entropy":       0.0,
            "segmented_url": None,
            "description":   None,
            "tips":          [],
            "precautions":   [],
            "sources":       [],
            "message":       "The image doesn't appear to be a focused close-up of a skin lesion. Please zoom in so the lesion fills most of the frame.",
        }

    # Upload the segmentation mask to Cloudinary
    segmented_url = _upload_mask_to_cloudinary(mask_full)

    # ── STAGE 2: 4-model ensemble ─────────────────────────────────────────────
    crop_t = infer_transforms(image=safe_crop)["image"].unsqueeze(0).to(device)
    # MaxViT requires 224×224
    crop_t_maxvit = F.interpolate(crop_t, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        p_c = F.softmax(_model_conv(crop_t),        dim=1).cpu().numpy()[0]
        p_r = F.softmax(_model_res(crop_t),         dim=1).cpu().numpy()[0]
        p_d = F.softmax(_model_den(crop_t),         dim=1).cpu().numpy()[0]
        p_m = F.softmax(_model_max(crop_t_maxvit),  dim=1).cpu().numpy()[0]

    # ── STAGE 3: Meta-boss ────────────────────────────────────────────────────
    combined = np.hstack([p_c, p_r, p_d, p_m]).reshape(1, -1)
    pred_idx      = int(_boss_model.predict(combined)[0])
    prob_dist     = _boss_model.predict_proba(combined)[0]
    confidence    = float(np.max(prob_dist) * 100)
    pred_entropy  = float(scipy_entropy(prob_dist, base=2))
    disease_code  = CLASSES[pred_idx]

    # ── STAGE 4: Confidence thresholds ────────────────────────────────────────
    #
    #  Tier              Condition                          Status
    #  ──────────────────────────────────────────────────────────────
    #  Healthy           confidence < 20%                  "healthy"
    #  Unknown           20% ≤ confidence < 60%            "unknown"
    #  Unknown           entropy > 1.8 (model confused)    "unknown"
    #  Low confidence    60% ≤ confidence < 80%            "low_confidence"
    #  Known             confidence ≥ 80%                  "known"
    #

    if confidence < 20.0:
        return {
            "status":        "healthy",
            "disease":       "No significant condition detected",
            "disease_code":  None,
            "confidence":    round(confidence, 1),
            "entropy":       round(pred_entropy, 3),
            "segmented_url": segmented_url,
            "description":   None,
            "tips":          [
                "Continue performing monthly self skin examinations.",
                "Apply SPF 30+ sunscreen daily to protect healthy skin.",
                "Maintain a healthy lifestyle — diet, hydration, and stress management all affect skin health.",
            ],
            "precautions":   [
                "Even healthy-looking skin should be checked annually by a dermatologist if you have risk factors.",
                "If you notice any new or changing lesions in the future, seek professional evaluation promptly.",
            ],
            "sources":       ["WHO Skin Cancer Guidelines 2023", "American Academy of Dermatology"],
            "message":       "The area appears healthy. No significant skin condition was detected.",
        }

    if confidence < 60.0 or pred_entropy > 1.8:
        return {
            "status":        "unknown",
            "disease":       "Unknown / Uncertain",
            "disease_code":  None,
            "confidence":    round(confidence, 1),
            "entropy":       round(pred_entropy, 3),
            "segmented_url": segmented_url,
            "description":   "The AI model was unable to confidently classify this lesion. This can happen when the condition is not among the four trained categories, when image quality is low, or when the lesion has atypical features.",
            "tips":          [
                "Try retaking the photo with better lighting and a closer zoom.",
                "Ensure the lesion is clean and in focus before scanning.",
                "Use dermoscopy images for more accurate results if available.",
            ],
            "precautions":   [
                "Please consult a board-certified dermatologist for a professional evaluation.",
                "Do not ignore any skin lesion that concerns you, regardless of AI output.",
                "Bring photos of the lesion taken at different times if possible.",
            ],
            "sources":       ["American Academy of Dermatology", "Mayo Clinic"],
            "message":       f"Uncertain result (confidence: {round(confidence, 1)}%). The model could not reliably classify this lesion. Please consult a dermatologist.",
        }

    # 60%–79%: low confidence — show disease but with a warning
    if confidence < 80.0:
        info = DISEASE_INFO.get(disease_code, {})
        return {
            "status":        "low_confidence",
            "disease":       info.get("full_name", disease_code),
            "disease_code":  disease_code,
            "confidence":    round(confidence, 1),
            "entropy":       round(pred_entropy, 3),
            "segmented_url": segmented_url,
            "description":   info.get("description"),
            "tips":          info.get("tips", []),
            "precautions":   info.get("precautions", []),
            "sources":       info.get("sources", []),
            "message":       f"Low confidence result ({round(confidence, 1)}%). This prediction should be verified by a qualified dermatologist.",
        }

    # ≥ 80%: high confidence — full diagnosis
    info = DISEASE_INFO.get(disease_code, {})
    return {
        "status":        "known",
        "disease":       info.get("full_name", disease_code),
        "disease_code":  disease_code,
        "confidence":    round(confidence, 1),
        "entropy":       round(pred_entropy, 3),
        "segmented_url": segmented_url,
        "description":   info.get("description"),
        "tips":          info.get("tips", []),
        "precautions":   info.get("precautions", []),
        "sources":       info.get("sources", []),
        "message":       None,
    }


# ==============================================================================
# ROUTES — HEALTH CHECK
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "models_loaded": _models_loaded,
        "device":        str(device),
    })


# ==============================================================================
# ROUTES — AI PREDICT
# ==============================================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
        { "imageUrl": "https://...", "photoType": "phone" | "dermo" }

    photoType is optional — defaults to "phone" (safer 35% lesion-area limit).
    Use "dermo" for dermoscope images (85% limit).

    Returns the full diagnosis JSON described in the plan.
    """
    try:
        # Lazy-load models if startup load failed
        if not _models_loaded:
            load_ai_models()

        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        image_url  = data.get("imageUrl") or data.get("image_url")
        photo_type = data.get("photoType", "phone").lower()

        if not image_url:
            return jsonify({"error": "imageUrl is required"}), 400

        # ── Download image from Cloudinary (or any URL) ───────────────────────
        headers = {"User-Agent": "SkinSight/1.0"}
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()

        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Failed to decode image. Ensure the URL points to a valid JPG or PNG."}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ── Run full pipeline ─────────────────────────────────────────────────
        result = _run_pipeline(img_rgb, photo_type=photo_type)
        return jsonify(result), 200

    except requests.exceptions.Timeout:
        return jsonify({"error": "Image download timed out. Please try again."}), 408
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except RuntimeError as e:
        # Catches CUDA OOM and similar torch errors
        print(f"[SkinSight] RuntimeError in predict: {e}")
        return jsonify({"error": "Model inference failed. The server may be under high load — please try again."}), 500
    except Exception as e:
        print(f"[SkinSight] Unexpected error in predict: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": "An unexpected error occurred during analysis."}), 500


# ==============================================================================
# ROUTES — AUTH (unchanged from your original backend)
# ==============================================================================

@app.route("/api/check-email", methods=["POST"])
def check_email():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    try:
        # ✅ Wakes up Firebase before checking!
        get_firebase()
        firebase_auth.get_user_by_email(email)
        return jsonify({"exists": True})
    except firebase_admin.auth.UserNotFoundError:
        return jsonify({"exists": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/send-password-reset", methods=["POST"])
def send_password_reset():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    otp_code = data.get("otp_code", "")

    try:
        # ✅ Wakes up Firebase before checking!
        get_firebase()
        user = firebase_auth.get_user_by_email(email)
        name = user.display_name.split(" ")[0] if user.display_name else "User"

        html = get_password_reset_html(name, otp_code)
        _send_email(email, "SkinSight – Password Reset Code", html)
        return jsonify({"success": True})
    except firebase_admin.auth.UserNotFoundError:
        return jsonify({"error": "No account found with this email address."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# ROUTES — EMAIL / OTP  (unchanged from your original backend)
# ==============================================================================

def _send_email(to_email: str, subject: str, html_body: str):
    gmail_user = os.getenv("GMAIL_EMAIL")
    gmail_pass = os.getenv("GMAIL_PASSWORD")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"SkinSight <{gmail_user}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, to_email, msg.as_string())


@app.route("/api/send-otp", methods=["POST"])
def send_otp():
    data     = request.get_json()
    email    = data.get("email", "").strip()
    name     = data.get("name", "User")
    otp_code = data.get("otp_code", "")
    source   = data.get("source", "mobile")
    if not email or not otp_code:
        return jsonify({"error": "email and otp_code are required"}), 400
    try:
        html = get_otp_email_html(name, otp_code, source)
        _send_email(email, "SkinSight – Your Verification Code", html)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/api/send-email-change-otp", methods=["POST"])
def send_email_change_otp():
    data      = request.get_json()
    new_email = data.get("email", "").strip()
    otp_code  = data.get("otp_code", "")
    source    = data.get("source", "mobile")
    uid       = data.get("uid", "")
    name      = data.get("name", "User")

    # Fetch real first name from Firestore profile
    if uid:
        try:
            db   = get_firebase()
            snap = db.collection("users").document(uid).get()
            if snap.exists:
                profile = snap.to_dict()
                name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip() or name
        except Exception:
            pass

    if not new_email or not otp_code:
        return jsonify({"error": "email and otp_code are required"}), 400
    try:
        html = get_email_change_html(name, otp_code, new_email, source)
        _send_email(new_email, "SkinSight – Email Change Verification", html)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update-password", methods=["POST"])
def update_password():
    data         = request.get_json()
    email        = data.get("email", "").strip()
    new_password = data.get("new_password", "")
    if not email or not new_password:
        return jsonify({"error": "email and new_password are required"}), 400
    try:
        user = firebase_auth.get_user_by_email(email)
        firebase_auth.update_user(user.uid, password=new_password)
        return jsonify({"success": True})
    except firebase_admin.exceptions.FirebaseError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)