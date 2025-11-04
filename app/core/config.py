"""
multi-agent-fits-dev-02/app/core/config.py
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    # API Keys
    openai_api_key: str

    # Storage Directories paths (as strings from .env)
    fitsfiles_dir: str = "storage/fitsfiles"
    plots_dir: str = "storage/plots"
    plots_psd_dir: str = "storage/plots/psd"
    plots_powerlaw_dir: str = "storage/plots/power_law"
    plots_bendingpowerlaw_dir: str = "storage/plots/bending_power_law"

    # Upload limits
    max_upload_size: int = 536_870_912    # 500MB

    # Database settings
    database_url: str = "postgresql+asyncpg://fits_user:fits_password@localhost:5432"
    database_echo: bool = False  # Set to True for SQL debugging
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_recycle: int = 3600  # Recycle connections after 1 hour
    database_pool_pre_ping: bool = True  # Verify connections before using

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # AstroSage Service
    astrosage_base_url: str = "http://192.168.156.22:8080"
    astrosage_model: str = "astrosage"
    astrosage_timeout: int = 240  # seconds
    astrosage_max_retries: int = 3
    astrosage_retry_delay: int = 5  # seconds

    # Conversation settings
    conversation_history_limit: int = 10

    # Default LLM parameters
    astrosage_default_temperature: float = 0.2
    astrosage_default_max_tokens: int = 600
    astrosage_default_top_p: float = 0.95

    # Rewrite Agent configuration
    rewrite_model: str = "mini"           # "mini", "turbo", or "standard"
    rewrite_auto_upgrade: bool = True     # Auto-upgrade for complex queries
    rewrite_temperature: float = 0.3
    rewrite_max_tokens: int = 3000
    
    # Cost tracking (optional)
    enable_cost_tracking: bool = True
    monthly_budget_limit: float = 100.0   # USD

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

# Global settings instance
settings = Settings()

    
