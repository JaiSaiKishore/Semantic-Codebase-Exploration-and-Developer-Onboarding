"""
Test script to simulate a GitHub Webhook push event with HMAC SHA-256 signature verification.
Run this script while `server.py` is running to see the live server console event log (Figure A2.5).
"""
import os
import json
import hmac
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "my_super_secret_webhook_key")

payload = {
    "ref": "refs/heads/main",
    "repository": {
        "name": "flask-todo",
        "full_name": "patrickloeber/flask-todo",
        "clone_url": "https://github.com/patrickloeber/flask-todo.git"
    },
    "commits": [
        {
            "id": "7b8f9e1234567890abcdef1234567890abcdef12",
            "message": "feat(app): add input validation and update Todo model AST",
            "added": [],
            "modified": ["app.py"]
        }
    ]
}

body_bytes = json.dumps(payload).encode("utf-8")
signature = "sha256=" + hmac.new(SECRET.encode(), msg=body_bytes, digestmod=hashlib.sha256).hexdigest()

headers = {
    "Content-Type": "application/json",
    "X-GitHub-Event": "push",
    "X-Hub-Signature-256": signature
}

print("=" * 60)
print("  TRIGGERING GITHUB WEBHOOK (HMAC SHA-256 SIMULATION)")
print("=" * 60)
print(f"Target URL: {SERVER_URL}/webhook/github")
print(f"Event: push")
print(f"Repository: {payload['repository']['full_name']}")
print(f"Computed Signature: {signature}")
print("-" * 60)

try:
    response = requests.post(f"{SERVER_URL}/webhook/github", data=body_bytes, headers=headers, timeout=5)
    print(f"HTTP Status: {response.status_code}")
    print(f"Response Body: {response.text}")
    print("-" * 60)
    if response.status_code == 200:
        print("[SUCCESS] Webhook accepted and HMAC signature verified!")
        print("Check your server terminal window running server.py to view the")
        print("exact Figure A2.5: GitHub Webhook Event Log.")
    else:
        print(f"[FAILED] Webhook returned status {response.status_code}")
except Exception as e:
    print(f"[ERROR] Could not connect to {SERVER_URL}: {e}")
    print("Please make sure server.py is running in another terminal window.")
print("=" * 60)

