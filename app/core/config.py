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
    astrosage_base_url: str

    # Storage Directories paths (as strings from .env)
    fitsfiles_dir: str = "storage/fitsfiles"
    plots_dir: str = "storage/plots"
    plots_psd_dir: str = "storage/plots/psd"
    plots_powerlaw_dir: str = "storage/plots/power_law"
    plots_bendingpowerlaw_dir: str = "storage/plots/bending_power_law"

    # Upload limits
    max_upload_size: int = 536870912  # 500MB

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False
    )

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

    
