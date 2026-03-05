import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Blueprint, jsonify, request

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GMAIL_EMAIL, GMAIL_PASSWORD
from email_templates import get_email_change_html, get_otp_email_html, get_password_reset_html

emails_bp = Blueprint('emails', __name__)


# ── Shared email sender ───────────────────────────────────────────────────────
def send_email(to_email, subject, html_body):
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"]    = GMAIL_EMAIL
    message["To"]      = to_email
    message.attach(MIMEText(html_body, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
        server.sendmail(GMAIL_EMAIL, to_email, message.as_string())


# ── POST /api/send-otp ────────────────────────────────────────────────────────
@emails_bp.route('/api/send-otp', methods=['POST'])
def send_otp():
    try:
        data     = request.get_json()
        email    = data.get('email')
        name     = data.get('name', 'User')
        otp_code = data.get('otp_code')

        print(f"\n📧 Sending verification OTP to: {email} | Name: {name} | OTP: {otp_code}")

        if not email or not otp_code:
            return jsonify({'success': False, 'message': 'Missing email or OTP code'}), 400
        if not GMAIL_EMAIL or not GMAIL_PASSWORD:
            return jsonify({'success': False, 'message': 'Server configuration error'}), 500

        send_email(email, "SkinSight – Your Verification Code", get_otp_email_html(name, otp_code))
        print(f"✅ Verification email sent to {email}")
        return jsonify({'success': True, 'message': f'OTP sent to {email}'}), 200

    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail authentication failed")
        return jsonify({'success': False, 'message': 'Gmail authentication failed'}), 401
    except Exception as e:
        print(f"❌ send-otp error: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/send-password-reset ────────────────────────────────────────────
@emails_bp.route('/api/send-password-reset', methods=['POST'])
def send_password_reset():
    try:
        data     = request.get_json()
        email    = data.get('email')
        name     = data.get('name', 'User')
        otp_code = data.get('otp_code')

        print(f"\n🔐 Sending password reset OTP to: {email} | Name: {name} | OTP: {otp_code}")

        if not email or not otp_code:
            return jsonify({'success': False, 'message': 'Missing email or OTP code'}), 400
        if not GMAIL_EMAIL or not GMAIL_PASSWORD:
            return jsonify({'success': False, 'message': 'Server configuration error'}), 500

        send_email(email, "SkinSight – Password Reset Code", get_password_reset_html(name, otp_code))
        print(f"✅ Password reset email sent to {email}")
        return jsonify({'success': True, 'message': f'Password reset OTP sent to {email}'}), 200

    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail authentication failed")
        return jsonify({'success': False, 'message': 'Gmail authentication failed'}), 401
    except Exception as e:
        print(f"❌ send-password-reset error: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/send-email-change-otp ──────────────────────────────────────────
@emails_bp.route('/api/send-email-change-otp', methods=['POST'])
def send_email_change_otp():
    try:
        data      = request.get_json()
        new_email = data.get('email')
        name      = data.get('name', 'User')
        otp_code  = data.get('otp_code')

        print(f"\n📧 Sending email change OTP to: {new_email} | Name: {name} | OTP: {otp_code}")

        if not new_email or not otp_code:
            return jsonify({'success': False, 'message': 'Missing email or OTP code'}), 400
        if not GMAIL_EMAIL or not GMAIL_PASSWORD:
            return jsonify({'success': False, 'message': 'Server configuration error'}), 500

        send_email(new_email, "SkinSight – Email Change Verification", get_email_change_html(name, otp_code, new_email))
        print(f"✅ Email change OTP sent to {new_email}")
        return jsonify({'success': True, 'message': f'Email change OTP sent to {new_email}'}), 200

    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail authentication failed")
        return jsonify({'success': False, 'message': 'Gmail authentication failed'}), 401
    except Exception as e:
        print(f"❌ send-email-change-otp error: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500