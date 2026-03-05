import os
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv

load_dotenv()

# ── Gmail ─────────────────────────────────────────────────────────────────────
GMAIL_EMAIL    = os.getenv('GMAIL_EMAIL', '').strip()
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD', '').replace(' ', '').strip()

# ── Firebase Admin SDK ────────────────────────────────────────────────────────
SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    print("✅ Firebase Admin SDK initialized")

print("🚀 Flask backend starting...")
print(f"📧 Gmail Email: {GMAIL_EMAIL}")
print(f"🔑 Gmail Password: {'✅ Found (16 chars)' if len(GMAIL_PASSWORD) == 16 else '⚠️ Check password - should be 16 chars'}")