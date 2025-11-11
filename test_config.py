"""
Test configuration loading
Run: python test_config.py
"""

from app.core.config import settings

print("=" * 60)
print("Configuration Test")
print("=" * 60)

# Database
print(f"Database URL: {settings.database_url[:50]}...")
print(f"Database Pool Size: {settings.database_pool_size}")
print(f"Database Echo: {settings.database_echo}")

# Storage Paths
print(f"\nStorage Directories:")
print(f"  FITS Files: {settings.fits_path}")
print(f"  Plots: {settings.plots_path}")
print(f"  PSD Plots: {settings.psd_plots_path}")
print(f"  Power Law Plots: {settings.powerlaw_plots_path}")
print(f"  Bending Power Law Plots: {settings.bendingpowerlaw_plots_path}")

# JWT
print(f"\nJWT Settings:")
print(f"  Secret Key: {settings.secret_key[:20]}...")
print(f"  Algorithm: {settings.algorithm}")
print(f"  Token Expire: {settings.access_token_expire_hours} hours")

# Password
print(f"\nPassword Settings:")
print(f"  Min Length: {settings.password_min_length}")

# API Keys
print(f"\nAPI Keys:")
print(f"  OpenAI API Key: {settings.openai_api_key[:20]}...")

# AstroSage
print(f"\nAstroSage Settings:")
print(f"  Base URL: {settings.astrosage_base_url}")
print(f"  Model: {settings.astrosage_model}")
print(f"  Timeout: {settings.astrosage_timeout}s")
print(f"  Max Retries: {settings.astrosage_max_retries}")

# Rewrite Agent
print(f"\nRewrite Agent:")
print(f"  Model: {settings.rewrite_model}")
print(f"  Temperature: {settings.rewrite_temperature}")
print(f"  Max Tokens: {settings.rewrite_max_tokens}")

# Conversation
print(f"\nConversation:")
print(f"  History Limit: {settings.conversation_history_limit}")

# Upload
print(f"\nUpload:")
print(f"  Max Size: {settings.max_upload_size / 1024 / 1024:.0f} MB")

# CORS
print(f"\nCORS:")
print(f"  Origins: {settings.cors_origins}")

print("=" * 60)
print("✅ Configuration loaded successfully!")
print("=" * 60)

# Check if directories exist
print("\nDirectory Check:")
import os
dirs_to_check = [
    settings.fits_path,
    settings.plots_path,
    settings.psd_plots_path,
    settings.powerlaw_plots_path,
    settings.bendingpowerlaw_plots_path
]

for dir_path in dirs_to_check:
    exists = "✅" if os.path.exists(dir_path) else "❌"
    print(f"  {exists} {dir_path}")

print("=" * 60)