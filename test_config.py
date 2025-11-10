"""
Test configuration loading
Run: python test_config.py
"""

from app.core.config import settings

print("=" * 60)
print("Configuration Test")
print("=" * 60)

# Database
print(f"Database URL: {settings.database_url[:30]}...")

# App
print(f"App Name: {settings.app_name}")
print(f"Debug: {settings.debug}")

# JWT
print(f"Secret Key: {settings.secret_key[:20]}...")
print(f"Algorithm: {settings.algorithm}")
print(f"Token Expire: {settings.access_token_expire_hours} hours")

# Password
print(f"Min Password Length: {settings.password_min_length}")

print("=" * 60)
print("✅ Configuration loaded successfully!")