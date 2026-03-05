import smtplib
from dotenv import load_dotenv
import os

load_dotenv()

email = os.getenv('GMAIL_EMAIL', '').strip()
password = os.getenv('GMAIL_PASSWORD', '').replace(' ', '').strip()

print(f"📧 Email: {email}")
print(f"🔑 Password (no spaces): {'*' * len(password)}")
print(f"🔑 Password length: {len(password)} chars (should be 16)")

# Test port 465 (SMTP_SSL)
print("\n--- Testing Port 465 (SMTP_SSL) ---")
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        print("✅ Connected on port 465!")
        smtp.login(email, password)
        print("✅ Login successful on port 465!")
except Exception as e:
    print(f"❌ Port 465 failed: {e}")

    # Fallback: Test port 587 (STARTTLS)
    print("\n--- Testing Port 587 (STARTTLS) ---")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            print("✅ Connected on port 587!")
            smtp.login(email, password)
            print("✅ Login successful on port 587!")
    except Exception as e2:
        print(f"❌ Port 587 also failed: {e2}")
        print("\n💡 Fix: Go to https://myaccount.google.com/apppasswords and generate a fresh App Password")