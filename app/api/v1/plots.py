"""
app/api/v1/plots.py
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import logging

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/plots/{plot_type}/{filename}")
async def get_plot(plot_type: str, filename: str):
    """
    Serve plot images
    
    Args:
        plot_type: Type of plot (psd, power_law, bending_power_law)
        filename: Plot filename
    
    Returns:
        Image file
    """

    # Map plot type to directory
    plot_dirs = {
        'psd': settings.psd_plots_path,
        'power_law': settings.powerlaw_plots_path,
        'bending_power_law': settings.bendingpowerlaw_plots_path
    }

    plot_dir = plot_dirs.get(plot_type)

    if not plot_dir:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot type: {plot_type}"
        )
    
    # Build full path
    file_path = plot_dir / filename

    # Check if file exists
    if not file_path.exists():
        logger.error(f"Plot not found: {file_path}")
        raise HTTPException(
            status_code=404,
            detail=f"Plot not found: {filename}"
        )
    
    # Check if it's a file (not directory)
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot path"
        )
    
    # Serve the image
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename
    )