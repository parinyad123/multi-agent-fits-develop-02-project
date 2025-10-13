"""
multi-agent-fits-dev-02/app/tools/fits/loader.py
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from astropy.io import fits
import os

logger = logging.getLogger(__name__)


def validate_fits_structure(path: str) -> Dict[str, Any]:
    """
    Validate the structure of a FITS file and extract basic metadata.
    
    Args:
        path: Path to the FITS file
        
    Returns:
        Dictionary with metadata and validation results
        
    Raises:
        ValueError: If the file is not a valid FITS file
    """
    try:
        with fits.open(path) as hdul:
            # Extract basic information about the file
            n_hdus = len(hdul)
            
            # Get information about each HDU
            hdu_info = []
            has_data = False
            
            for i, hdu in enumerate(hdul):
                hdu_type = type(hdu).__name__
                
                # Extract header metadata
                cards = []
                for card in hdu.header.cards:
                    if card.keyword not in ['COMMENT', 'HISTORY'] and card.keyword:
                        cards.append({
                            'keyword': card.keyword,
                            'value': str(card.value),
                            'comment': card.comment
                        })
                
                # Check if this HDU has data
                shape = None
                data_type = None
                n_rows = None
                column_names = []
                
                if hdu.data is not None:
                    has_data = True
                    
                    if isinstance(hdu, fits.BinTableHDU):
                        # Binary table data
                        n_rows = hdu.data.shape[0]
                        column_names = hdu.columns.names
                        data_type = "binary_table"
                    elif isinstance(hdu, fits.TableHDU):
                        # ASCII table data
                        n_rows = hdu.data.shape[0]
                        column_names = hdu.columns.names
                        data_type = "ascii_table"
                    else:
                        # Image data
                        shape = hdu.data.shape
                        data_type = str(hdu.data.dtype)
                
                hdu_info.append({
                    'hdu_index': i,
                    'hdu_type': hdu_type,
                    'header_cards': cards,
                    'data_present': hdu.data is not None,
                    'data_type': data_type,
                    'shape': shape,
                    'n_rows': n_rows,
                    'column_names': column_names
                })
            
            # Verify this is actually a FITS file
            if n_hdus == 0:
                raise ValueError("Empty or invalid FITS file")
                
            if not has_data:
                logger.warning(f"FITS file {path} has no data")
            
            # Return metadata
            return {
                'n_hdus': n_hdus,
                'has_data': has_data,
                'hdu_info': hdu_info
            }
    
    except Exception as e:
        logger.error(f"Error validating FITS file {path}: {str(e)}")
        raise ValueError(f"Invalid FITS file: {str(e)}")


def get_fits_header(path: str) -> dict:
    """
    Extract header information from a FITS file.
    
    Args:
        path: Path to the FITS file
        
    Returns:
        Dictionary with header information
    """
    try:
        with fits.open(path) as hdul:
            primary_header = hdul[0].header

            # Extract the XDAL0 value if it exists
            xdal0 = primary_header.get('XDAL0', '')

            # If XDAL0 exists, parse the filename
            filename = ''
            if xdal0:
                # XDAL0 format example: '0792180601_PN_source_lc_300_1000eV.fits 2024-04-20T15:08:06.000 Cre&'
                parts = xdal0.split()
                if parts:
                    filename = parts[0]  # Get the first part which should be the filename
            
            return {
                "filename": filename or os.path.basename(path),
                "xdal0": xdal0,
                "header": {k: str(v) for k, v in primary_header.items()}
            }
    except Exception as e:
        logger.error(f"Error extracting header from FITS file {path}: {str(e)}")
        return {
            "filename": os.path.basename(path), 
            "error": str(e)
        }
    
def load_fits_data(path: str, hdu_index: int = 1, column: str = 'Rate') -> np.ndarray:
    """
    Load data from a FITS file.
    
    Args:
        path: Path to the FITS file
        hdu_index: Index of the HDU to load data from (default: 1)
        column: Column name to extract (for table HDUs)
        
    Returns:
        NumPy array with the extracted data
        
    Raises:
        ValueError: If the file cannot be loaded or the data cannot be extracted
    """
    try:
        with fits.open(path) as hdul:
            if hdu_index >= len(hdul):
                raise ValueError(f"HDU index {hdu_index} out of range (file has {len(hdul)} HDUs)")
            
            hdu = hdul[hdu_index]

            if hdu.data is None:
                raise ValueError(f"HDU {hdu_index} has no data")
            
            # Extract data based on HDU type
            if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                # Case-insensitive column matching
                available_columns = hdu.columns.names

                # Try exact match first
                if column in available_columns:
                    data = hdu.data[column]
                else:
                    # Try case-insensitive match
                    column_lower = column.lower()
                    matching_columns = [col for col in available_columns if col.lower() == column_lower]
                    
                    if matching_columns:
                        data = hdu.data[matching_columns[0]]
                    else:
                        # No match found
                        raise ValueError(
                            f"Column '{column}' not found in HDU {hdu_index}. "
                            f"Available columns: {', '.join(available_columns)}"
                        )
            else:
                # Image data
                data = hdu.data
            
            # Convert to numpy array and clean
            array_data = np.array(data)
            clean_data = np.nan_to_num(array_data, nan=0.0)
            
            # Ensure non-negative values for time series
            return np.where(clean_data < 0, 0, clean_data)

    except Exception as e:
        logger.error(f"Error loading FITS file {path}: {str(e)}")
        raise ValueError(f"Failed to load FITS data: {str(e)}")