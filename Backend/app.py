import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

import config  # noqa: F401 — initializes Firebase + loads env vars
from routes.auth import auth_bp
from routes.emails import emails_bp

app = Flask(__name__)
CORS(app)

# ── Register blueprints ───────────────────────────────────────────────────────
app.register_blueprint(auth_bp)
app.register_blueprint(emails_bp)


# ── GET /api/health ───────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'ok',
        'message':   'Flask backend is running',
        'timestamp': datetime.now().isoformat(),
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Flask server starting on http://0.0.0.0:5000")
    print("Access from browser: http://localhost:5000")
    print("Access from phone:   http://192.168.100.2:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)