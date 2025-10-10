"""
multi-agent-fits-dev-02/app/utils/file_manager.py
(Updated version with better error handling)
"""

from pathlib import Path
from typing import Optional
import shutil
import logging
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Utility class for managing FITS files and plots"""
    
    @staticmethod
    def save_fits_file(file_id: str, file_content: bytes) -> Path:
        """
        Save uploaded FITS file
        
        Args:
            file_id: Unique file identifier
            file_content: File content as bytes
            
        Returns:
            Path to saved file
            
        Raises:
            IOError: If file cannot be saved
        """
        try:
            # Ensure directory exists
            settings.fits_path.mkdir(parents=True, exist_ok=True)
            
            file_path = settings.fits_path / f"{file_id}.fits"
            file_path.write_bytes(file_content)
            logger.info(f"FITS file saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving FITS file {file_id}: {e}")
            raise IOError(f"Failed to save file: {str(e)}")
    
    @staticmethod
    def get_fits_file_path(file_id: str) -> Optional[Path]:
        """
        Get path to FITS file if it exists
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Path if file exists, None otherwise
        """
        file_path = settings.fits_path / f"{file_id}.fits"
        return file_path if file_path.exists() else None
    
    @staticmethod
    def delete_fits_file(file_id: str) -> bool:
        """
        Delete FITS file from filesystem
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            True if deleted, False if file doesn't exist
        """
        file_path = settings.fits_path / f"{file_id}.fits"
        
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"FITS file deleted: {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error deleting file {file_id}: {e}")
                raise IOError(f"Failed to delete file: {str(e)}")
        
        return False

    @staticmethod
    def save_plot(plot_id: str, plot_type: str, plot_data: bytes) -> Path:
        """
        Save generated plot
        
        Args:
            plot_id: Unique identifier for the plot
            plot_type: Type of plot ('psd', 'power_law', 'bending_power_law')
            plot_data: PNG image data
            
        Returns:
            Path to saved plot
            
        Raises:
            ValueError: If plot_type is invalid
            IOError: If plot cannot be saved
        """
        plot_dirs = {
            'psd': settings.psd_plots_path,
            'power_law': settings.powerlaw_plots_path,
            'bending_power_law': settings.bendingpowerlaw_plots_path
        }

        plot_dir = plot_dirs.get(plot_type)
        if not plot_dir:
            raise ValueError(f"Invalid plot type: {plot_type}")

        try:
            # Ensure directory exists
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = plot_dir / f"{plot_type}_{plot_id}.png"
            file_path.write_bytes(plot_data)
            logger.info(f"Plot saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving plot {plot_id}: {e}")
            raise IOError(f"Failed to save plot: {str(e)}")

    @staticmethod
    def get_plot_path(plot_id: str, plot_type: str) -> Optional[Path]:
        """Get path to plot if it exists"""
        plot_dirs = {
            'psd': settings.psd_plots_path,
            'power_law': settings.powerlaw_plots_path,
            'bending_power_law': settings.bendingpowerlaw_plots_path
        }
        
        plot_dir = plot_dirs.get(plot_type)
        if not plot_dir:
            return None
        
        file_path = plot_dir / f"{plot_type}_{plot_id}.png"
        return file_path if file_path.exists() else None

    @staticmethod
    def cleanup_old_files(days_old: int = 30) -> tuple[int, int]:
        """
        Clean up files older than specified days
        
        NOTE: Currently not used automatically
        Can be called manually or via background job
        
        Args:
            days_old: Number of days to keep files
            
        Returns:
            Tuple of (fits_files_deleted, plots_deleted)
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        fits_deleted = 0
        plots_deleted = 0
        
        # Clean FITS files
        for file_path in settings.fits_path.glob("*.fits"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                fits_deleted += 1
        
        # Clean plots
        for plot_dir in [settings.psd_plots_path, 
                        settings.powerlaw_plots_path, 
                        settings.bendingpowerlaw_plots_path]:
            for file_path in plot_dir.glob("*.png"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    plots_deleted += 1
        
        logger.info(f"Cleanup completed: {fits_deleted} FITS files, {plots_deleted} plots deleted")
        return fits_deleted, plots_deleted