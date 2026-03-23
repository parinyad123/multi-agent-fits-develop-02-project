"""
Application Configuration
Environment variables and settings

multi-agent-fits-dev-02/app/core/config.py
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import secrets

from app.core.constants import ModelNames


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    # ==========================================
    # API Keys
    # ==========================================
    openai_api_key: str = ""   # legacy — no longer required
    groq_api_key: str = ""

    # ==========================================
    # Storage Directories (relative paths from .env)
    # ==========================================
    fitsfiles_dir: str = "storage/fitsfiles"
    plots_dir: str = "storage/plots"
    plots_psd_dir: str = "storage/plots/psd"
    plots_powerlaw_dir: str = "storage/plots/power_law"
    plots_bendingpowerlaw_dir: str = "storage/plots/bending_power_law"

    # ==========================================
    # Upload Limits
    # ==========================================
    max_upload_size: int = 10240  # 10MB

    # ==========================================
    # Database Settings
    # ==========================================
    database_url: str = "postgresql+asyncpg://fits_user:fits_password@localhost:5433/fits_analysis_db"
    database_echo: bool = False  # Set to True for SQL debugging
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_recycle: int = 3600  # Recycle connections after 1 hour
    database_pool_pre_ping: bool = True  # Verify connections before using

    # ==========================================
    # Model per agent
    # ==========================================
    # Classification agent  — needs structured JSON output, fast
    CLASSIFICATION_MODEL: str = ModelNames.LLAMA33_70B

    # Interpretation agent  — replaces AstroSage, needs deep reasoning
    # gpt-oss works great here; fall back to ModelNames.LLAMA33_70B
    INTERPRETATION_MODEL: str = ModelNames.GPT_OSS_120B

    # Rewrite agent  — formats final response to user
    REWRITE_MODEL: str = ModelNames.LLAMA33_70B

    # ==========================================
    # Groq Service (shared transport settings)
    # ==========================================
    groq_model: str = ModelNames.LLAMA33_70B  # default fallback
    groq_timeout: int = 60   # seconds
    groq_max_retries: int = 3
    groq_retry_delay: int = 5  # seconds

    # ==========================================
    # Conversation Settings
    # ==========================================
    conversation_history_limit: int = 10

    # ==========================================
    # Default LLM Parameters
    # ==========================================
    groq_default_temperature: float = 0.2
    groq_default_max_tokens: int = 4096

    # ==========================================
    # Rewrite Agent Configuration
    # ==========================================
    rewrite_model: str = "mini"  # "mini", "turbo", or "standard"
    rewrite_auto_upgrade: bool = True  # Auto-upgrade for complex queries
    rewrite_temperature: float = 0.3
    rewrite_max_tokens: int = 3000
    
    # ==========================================
    # Cost Tracking (optional)
    # ==========================================
    enable_cost_tracking: bool = True
    monthly_budget_limit: float = 100.0  # USD

    # ==========================================
    # CORS Settings
    # ==========================================
    cors_origins: list[str] = ["*"]

    # ==========================================
    # JWT Settings (NEW!)
    # ==========================================
    secret_key: str = "rohwLLiHvNZT-H6_fsxfDCZyx6ta_Da7S1PbbNVnuMc"  # Will be overridden by .env
    algorithm: str = "HS256"
    access_token_expire_hours: int = 8

    # ==========================================
    # Password Settings (NEW!)
    # ==========================================
    password_min_length: int = 8

    # ==========================================
    # Pydantic v2 Configuration
    # ==========================================
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Ignore extra fields in .env
    )

    # ==========================================
    #  Prompt log directory
    # ==========================================
    prompt_log_dir: Optional[str] = None


    # ==========================================
    # Properties (Directory Paths)
    # ==========================================
    
    @property
    def project_root(self) -> Path:
        """Get the root directory of the project."""
        return Path(__file__).parent.parent.parent

    @property
    def fits_path(self) -> Path:
        """Get absolute path to the FITS files directory."""
        return self.project_root / self.fitsfiles_dir

    @property
    def plots_path(self) -> Path:
        """Get absolute path to plots directory"""
        return self.project_root / self.plots_dir
    
    @property
    def psd_plots_path(self) -> Path:
        """Get absolute path to PSD plots directory"""
        return self.project_root / self.plots_psd_dir

    @property
    def powerlaw_plots_path(self) -> Path:
        """Get absolute path to power law plots directory"""
        return self.project_root / self.plots_powerlaw_dir

    @property 
    def bendingpowerlaw_plots_path(self) -> Path:
        """Get absolute path to bending power law plots directory"""
        return self.project_root / self.plots_bendingpowerlaw_dir

    # ==========================================
    # Helper Methods
    # ==========================================
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist"""
        directories = [
            self.fits_path,
            self.plots_path,
            self.psd_plots_path,
            self.powerlaw_plots_path,
            self.bendingpowerlaw_plots_path
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# ==========================================
# Global Settings Instance
# ==========================================

settings = Settings()

# Ensure directories exist on startup
settings.ensure_directories()