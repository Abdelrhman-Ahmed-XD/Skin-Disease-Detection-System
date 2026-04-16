# email_templates.py
# All HTML email templates for SkinSight emails

def get_header_html():
    return """
      <div style="background-color: #004F7F; padding: 40px 20px; text-align: center; width: 100%; box-sizing: border-box;">
        <div style="font-size: 48px; font-weight: bold; color: #ffffff; letter-spacing: 2px; font-family: Georgia, serif; line-height: 1.2;">
          <span style="color: #00A3A3; font-size: 56px;">S</span>kinsight
        </div>
        <div style="color: #C5E3ED; font-size: 14px; margin-top: 6px; font-style: italic; letter-spacing: 3px; font-family: Georgia, serif;">
          Snap. Detect. Protect.
        </div>
        <div style="width: 60px; height: 3px; background-color: #00A3A3; margin: 15px auto 0 auto; border-radius: 10px;"></div>
      </div>
      <div style="background-color: #00A3A3; padding: 12px 20px; text-align: center; width: 100%; box-sizing: border-box;">
        <p style="color: #ffffff; font-size: 14px; margin: 0; font-style: italic; letter-spacing: 0.5px; font-family: Georgia, serif;">Your skin health companion</p>
      </div>"""

def get_footer_html():
    return """
      <div style="background-color: #004F7F; padding: 30px 20px; text-align: center; width: 100%; box-sizing: border-box;">
        <div style="width: 40px; height: 2px; background-color: #00A3A3; margin: 0 auto 20px auto; border-radius: 10px;"></div>
        <div style="font-size: 22px; font-weight: bold; color: #ffffff; font-family: Georgia, serif; margin-bottom: 5px;"><span style="color: #00A3A3;">S</span>kinsight</div>
        <p style="color: #C5E3ED; font-size: 12px; margin: 8px 0 0 0; font-family: system-ui, sans-serif;">© 2026 SkinSight. All rights reserved.</p>
        <p style="color: #8ab4c9; font-size: 11px; margin-top: 6px; font-family: system-ui, sans-serif;">This is an automated message, please do not reply directly to this email.</p>
        <p style="color: #8ab4c9; font-size: 11px; margin-top: 5px; font-family: system-ui, sans-serif;">📧 skinsight.help.2025@gmail.com</p>
      </div>"""

def get_otp_block_html(otp_code, label="YOUR VERIFICATION CODE"):
    return f"""
        <div style="background-color: #D8E9F0; border-radius: 16px; padding: 30px 15px; text-align: center; border: 2px solid #C5E3ED; margin: 20px 0; width: 100%; box-sizing: border-box;">
          <div style="color: #004F7F; font-size: 12px; font-weight: bold; letter-spacing: 2px; margin-bottom: 20px; font-family: system-ui, sans-serif;">{label}</div>
          <div style="background-color: #004F7F; border-radius: 12px; padding: 18px 25px; display: inline-block; margin: 0 auto;">
            <span style="font-size: 42px; font-weight: bold; color: #ffffff; letter-spacing: 10px; font-family: Georgia, serif;">{otp_code}</span>
          </div>
          <div style="margin-top: 18px;">
            <span style="background-color: #00A3A3; border-radius: 20px; padding: 6px 18px; color: #ffffff; font-size: 12px; font-weight: 600; font-family: system-ui, sans-serif;">⏱ Expires in 10 minutes</span>
          </div>
        </div>"""

def get_security_notice_html(message):
    return f"""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px; padding: 14px 16px; margin-top: 20px; width: 100%; box-sizing: border-box;">
          <p style="font-size: 12px; color: #856404; margin: 0; line-height: 1.6; font-family: system-ui, sans-serif;">
            ⚠️ <strong>Security Notice:</strong> {message}
          </p>
        </div>"""


# ── Email Verification OTP ────────────────────────────────────────────────────
def get_otp_email_html(name, otp_code, source="mobile"):
    platform_text = "website" if source == "web" else "app"
    return f"""
    <div style="font-family: Georgia, serif; max-width: 600px; width: 100%; margin: 0 auto; background-color: #D8E9F0; border-radius: 16px; overflow: hidden; box-sizing: border-box;">
      {get_header_html()}
      <div style="background-color: #ffffff; padding: 35px 25px; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; width: 100%; box-sizing: border-box;">
        <p style="font-size: 20px; color: #004F7F; font-weight: bold; margin-bottom: 5px; font-family: Georgia, serif;">Hello, {name} 👋</p>
        <p style="font-size: 14px; color: #6B7280; margin-top: 0; line-height: 1.6; font-family: system-ui, sans-serif;">
          We received a request to verify your email address for your <strong style="color: #004F7F;">SkinSight</strong> account. Please use the code below to complete your verification.
        </p>
        <div style="height: 1px; background-color: #E5E7EB; margin: 20px 0;"></div>
        {get_otp_block_html(otp_code, "YOUR VERIFICATION CODE")}
        <div style="background-color: #f9fafb; border-radius: 12px; padding: 20px; margin: 20px 0; border-left: 4px solid #00A3A3; width: 100%; box-sizing: border-box;">
          <p style="font-size: 13px; color: #374151; font-weight: bold; margin: 0 0 12px 0; font-family: system-ui, sans-serif;">How to use your code:</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">1️⃣ Go back to the SkinSight {platform_text}</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">2️⃣ Enter the 6-digit code in the verification screen</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">3️⃣ Complete your registration and start using SkinSight!</p>
        </div>
        {get_security_notice_html("Do not share this code with anyone. SkinSight will never ask for your verification code. If you did not request this, please ignore this email.")}
      </div>
      {get_footer_html()}
    </div>"""


# ── Password Reset OTP ────────────────────────────────────────────────────────
def get_password_reset_html(name, otp_code, source="mobile"):
    platform_text = "website" if source == "web" else "app"
    return f"""
    <div style="font-family: Georgia, serif; max-width: 600px; width: 100%; margin: 0 auto; background-color: #D8E9F0; border-radius: 16px; overflow: hidden; box-sizing: border-box;">
      {get_header_html()}
      <div style="background-color: #ffffff; padding: 35px 25px; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; width: 100%; box-sizing: border-box;">
        <p style="font-size: 20px; color: #004F7F; font-weight: bold; margin-bottom: 5px; font-family: Georgia, serif;">Hello, {name} 👋</p>
        <p style="font-size: 14px; color: #6B7280; margin-top: 0; line-height: 1.6; font-family: system-ui, sans-serif;">
          We received a request to <strong style="color: #004F7F;">reset your password</strong> for your <strong style="color: #004F7F;">SkinSight</strong> account. Use the code below to reset your password.
        </p>
        <div style="height: 1px; background-color: #E5E7EB; margin: 20px 0;"></div>
        {get_otp_block_html(otp_code, "YOUR PASSWORD RESET CODE")}
        <div style="background-color: #f9fafb; border-radius: 12px; padding: 20px; margin: 20px 0; border-left: 4px solid #00A3A3; width: 100%; box-sizing: border-box;">
          <p style="font-size: 13px; color: #374151; font-weight: bold; margin: 0 0 12px 0; font-family: system-ui, sans-serif;">How to reset your password:</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">1️⃣ Go back to the SkinSight {platform_text}</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">2️⃣ Enter the 6-digit code on the reset screen</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">3️⃣ Create your new password and log in!</p>
        </div>
        {get_security_notice_html("If you did not request a password reset, please ignore this email. Your account is safe. Never share this code with anyone — SkinSight will never ask for it.")}
      </div>
      {get_footer_html()}
    </div>"""


# ── Email Change OTP ──────────────────────────────────────────────────────────
def get_email_change_html(name, otp_code, new_email, source="mobile"):
    platform_text = "website" if source == "web" else "app"
    return f"""
    <div style="font-family: Georgia, serif; max-width: 600px; width: 100%; margin: 0 auto; background-color: #D8E9F0; border-radius: 16px; overflow: hidden; box-sizing: border-box;">
      {get_header_html()}
      <div style="background-color: #ffffff; padding: 35px 25px; border-left: 1px solid #C5E3ED; border-right: 1px solid #C5E3ED; width: 100%; box-sizing: border-box;">
        <p style="font-size: 20px; color: #004F7F; font-weight: bold; margin-bottom: 5px; font-family: Georgia, serif;">Hello, {name} 👋</p>
        <p style="font-size: 14px; color: #6B7280; margin-top: 0; line-height: 1.6; font-family: system-ui, sans-serif;">
          We received a request to <strong style="color: #004F7F;">change the email address</strong> linked to your <strong style="color: #004F7F;">SkinSight</strong> account.
          Your new email will be: <strong style="color: #004F7F;">{new_email}</strong>
        </p>
        <div style="height: 1px; background-color: #E5E7EB; margin: 20px 0;"></div>
        {get_otp_block_html(otp_code, "YOUR EMAIL CHANGE CODE")}
        <div style="background-color: #f9fafb; border-radius: 12px; padding: 20px; margin: 20px 0; border-left: 4px solid #00A3A3; width: 100%; box-sizing: border-box;">
          <p style="font-size: 13px; color: #374151; font-weight: bold; margin: 0 0 12px 0; font-family: system-ui, sans-serif;">How to confirm your new email:</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">1️⃣ Go back to the SkinSight {platform_text}</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">2️⃣ Enter the 6-digit code in the verification screen</p>
          <p style="font-size: 13px; color: #6B7280; margin: 8px 0; font-family: system-ui, sans-serif; line-height: 1.5;">3️⃣ Your email will be updated to <strong>{new_email}</strong></p>
        </div>
        {get_security_notice_html("If you did not request an email change, please ignore this email and your account email will remain unchanged. Never share this code with anyone.")}
      </div>
      {get_footer_html()}
    </div>"""


def get_header_html():
    # Use the Cloudinary link you provided
    logo_url = "https://res.cloudinary.com/dignpxpgy/image/upload/v1776298242/WhatsApp_Image_2026-04-16_at_12.57.04_AM_uwqizy.jpg"

    return f"""
      <div style="background-color: #004F7F; padding: 30px 20px; text-align: center; width: 100%; box-sizing: border-box;">

        <table align="center" border="0" cellpadding="0" cellspacing="0" style="margin: 0 auto 15px auto;">
          <tr>
            <td align="center" style="background-color: #ffffff; border: 3px solid #00A3A3; border-radius: 50%; overflow: hidden; width: 80px; height: 80px;">
                <img src="{logo_url}" 
                     alt="SkinSight Logo" 
                     width="80" 
                     height="80" 
                     style="display: block; width: 80px; height: 80px; border-radius: 50%; object-fit: cover;"
                />
            </td>
          </tr>
        </table>

        <div style="font-size: 42px; font-weight: bold; color: #ffffff; letter-spacing: 2px; font-family: Georgia, serif; line-height: 1.2;">
          <span style="color: #00A3A3; font-size: 50px;">S</span>kinsight
        </div>
        <div style="color: #C5E3ED; font-size: 14px; margin-top: 6px; font-style: italic; letter-spacing: 3px; font-family: Georgia, serif;">
          Snap. Detect. Protect.
        </div>
        <div style="width: 60px; height: 3px; background-color: #00A3A3; margin: 15px auto 0 auto; border-radius: 10px;"></div>
      </div>
      <div style="background-color: #00A3A3; padding: 12px 20px; text-align: center; width: 100%; box-sizing: border-box;">
        <p style="color: #ffffff; font-size: 14px; margin: 0; font-style: italic; letter-spacing: 0.5px; font-family: Georgia, serif;">Your skin health companion</p>
      </div>"""