import traceback

from firebase_admin import auth as firebase_auth
from flask import Blueprint, jsonify, request

auth_bp = Blueprint('auth', __name__)


# ── POST /api/check-email ─────────────────────────────────────────────────────
@auth_bp.route('/api/check-email', methods=['POST'])
def check_email():
    """Check if an email is already registered in Firebase"""
    try:
        data  = request.get_json()
        email = data.get('email', '').strip()

        if not email:
            return jsonify({'success': False, 'exists': False, 'message': 'Missing email'}), 400

        print(f"\n🔍 Checking if email exists in Firebase: {email}")

        try:
            firebase_auth.get_user_by_email(email)
            print(f"⚠️ Email already in use: {email}")
            return jsonify({'success': True, 'exists': True}), 200
        except firebase_auth.UserNotFoundError:
            print(f"✅ Email is available: {email}")
            return jsonify({'success': True, 'exists': False}), 200

    except Exception as e:
        print(f"❌ check-email error: {e}")
        return jsonify({'success': False, 'exists': False, 'message': str(e)}), 500


# ── POST /api/update-email ────────────────────────────────────────────────────
@auth_bp.route('/api/update-email', methods=['POST'])
def update_email():
    """Update user email immediately using Firebase Admin SDK"""
    try:
        data      = request.get_json()
        uid       = data.get('uid', '').strip()
        new_email = data.get('new_email', '').strip()

        print(f"\n📧 Updating email for UID: {uid} → {new_email}")

        if not uid or not new_email:
            return jsonify({'success': False, 'message': 'Missing uid or new_email'}), 400

        # Check email isn't already taken by a different account
        try:
            existing = firebase_auth.get_user_by_email(new_email)
            if existing.uid != uid:
                print(f"⚠️ Email already in use by another account: {new_email}")
                return jsonify({'success': False, 'message': 'Email already in use by another account'}), 409
        except firebase_auth.UserNotFoundError:
            pass  # Email is free, proceed

        firebase_auth.update_user(uid, email=new_email)
        print(f"✅ Email updated successfully to: {new_email}")
        return jsonify({'success': True, 'message': 'Email updated successfully'}), 200

    except firebase_auth.UserNotFoundError:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    except Exception as e:
        print(f"❌ update-email error: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


# ── POST /api/update-password ─────────────────────────────────────────────────
@auth_bp.route('/api/update-password', methods=['POST'])
def update_password():
    """Update user password using Firebase Admin SDK — no old password needed"""
    try:
        data         = request.get_json()
        email        = data.get('email')
        new_password = data.get('new_password')

        print(f"\n🔐 Updating password for: {email}")

        if not email or not new_password:
            return jsonify({'success': False, 'message': 'Missing email or new password'}), 400

        user = firebase_auth.get_user_by_email(email)
        print(f"✅ Found user: {user.uid}")

        firebase_auth.update_user(user.uid, password=new_password)
        print(f"✅ Password updated successfully for {email}")
        return jsonify({'success': True, 'message': 'Password updated successfully'}), 200

    except firebase_auth.UserNotFoundError:
        print(f"❌ User not found: {email}")
        return jsonify({'success': False, 'message': 'User not found'}), 404
    except Exception as e:
        print(f"❌ update-password error: {e}"); traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500