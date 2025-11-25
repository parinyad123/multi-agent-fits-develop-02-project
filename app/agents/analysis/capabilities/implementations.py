
"""
multi-agent-fits-dev-02/app/agents/analysis/capabilities/implementations.py

Concrete implementations of analysis capabilities
"""

from typing import Dict, Any, Optional, Tuple
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
import io
import os

from app.core.config import settings
from app.core.constants import AnalysisType
from app.agents.analysis.capabilities.base import AnalysisCapability
from app.tools.statistics import calculate_statistics
from app.tools.psd import compute_psd, bin_psd
from app.tools.fitting import fit_power_law, fit_bending_power_law
from app.tools.plotting import  (
    plot_bending_power_law_with_residual_figure,
    plot_power_law_with_residual_figure,
    plot_psd_figure
)
from app.utils.file_manager import FileManager

# Import scipy for distribution summary
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)
# ==========================================
# Statistics Capability
# ==========================================

class StatisticsCapability(AnalysisCapability):
    """Calculate statistical metrics from time series data"""

    def __init__(self):
        super().__init__("statistics")
    
    async def execute(
        self, 
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        
        """
        Calculate statistics including percentiles and quantiles
        
        Parameters expected (with defaults from Classification Agent):
        {
            "metrics": ["mean", "median", "std", "min", "max", "count"],
            "percentiles": [25, 50, 75, 90, 95, 99],  # Optional
            "quantiles": [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]  # Optional
        }
        
        Note: percentiles and quantiles are related:
        - percentile_25 = quantile_0.25
        - percentile_50 = median = quantile_0.5
        - percentile_75 = quantile_0.75
        """
        self.logger.info(f"Executing statistics analysis with {rate_data.size} data points")

        
        # Extract parameters
        metrics = parameters.get("metrics", ["mean", "median", "std", "min", "max", "count"])
        percentiles = parameters.get("percentiles", [])
        quantiles = parameters.get("quantiles", [])

        # Calculate basic statistics (no validation needed)
        stats = calculate_statistics(rate_data, metrics)
        
        # Calculate percentiles
        if percentiles:
            self.logger.info(f"Computing percentiles: {percentiles}")
            for p in percentiles:
                if 0 <= p <= 100:
                    stats[f"percentile_{p}"] = float(np.percentile(rate_data, p))
        
        # Calculate quantiles
        if quantiles:
            self.logger.info(f"Computing quantiles: {quantiles}")
            for q in quantiles:
                if 0 <= q <= 1:
                    q_key = f"quantile_{str(q).replace('.', '_')}"
                    stats[q_key] = float(np.quantile(rate_data, q))
        
        # Add distribution summary
        if percentiles or quantiles:
            stats["distribution_summary"] = self._create_distribution_summary(rate_data, stats)
        
        result = {
            "statistics": stats,
            "n_data_points": int(rate_data.size),
            "parameters_used": parameters
        }
        
        self.logger.info(f"Statistics completed: {len(stats)} metrics")
        
        return (result, None)
    
    def _create_distribution_summary(
        self, 
        rate_data: np.ndarray, 
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create distribution summary"""
        
        summary = {}
        
        # Range
        if "min" in stats and "max" in stats:
            summary["range"] = {
                "min": stats["min"],
                "max": stats["max"],
                "span": stats["max"] - stats["min"]
            }
        
        # IQR
        q25_key = "percentile_25" if "percentile_25" in stats else "quantile_0_25"
        q75_key = "percentile_75" if "percentile_75" in stats else "quantile_0_75"
        
        if q25_key in stats and q75_key in stats:
            q25, q75 = stats[q25_key], stats[q75_key]
            iqr = q75 - q25
            summary["iqr"] = {
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "lower_fence": q25 - 1.5 * iqr,
                "upper_fence": q75 + 1.5 * iqr
            }
        
        # Coefficient of variation
        if "mean" in stats and "std" in stats and stats["mean"] != 0:
            summary["coefficient_of_variation"] = stats["std"] / abs(stats["mean"])
        
        # Skewness and Kurtosis
        summary["skewness"] = float(scipy_stats.skew(rate_data))
        summary["kurtosis"] = float(scipy_stats.kurtosis(rate_data))
        
        return summary

# ==========================================
# PSD Capability
# ==========================================

class PSDCapability(AnalysisCapability):
    """Compute Power Spectral Density"""
    
    def __init__(self):
        super().__init__("psd")
    
    async def execute(
        self, 
        rate_data: np.ndarray,  # Pre-validated
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Compute PSD and generate plot"""
        
        self.logger.info(f"Computing PSD with {rate_data.size} data points")
        
        # Extract parameters
        low_freq = parameters.get("low_freq", 1e-5)
        high_freq = parameters.get("high_freq", 0.05)
        bins = parameters.get("bins", 3500)
        filename = parameters.get('filename', 'Unknown')
        
        self.logger.info(f"Computing PSD with {len(rate_data)} data points")
        
        # Compute PSD (no validation needed)
        freqs, psd = compute_psd(rate_data)
        x, y = bin_psd(freqs, psd, low_freq, high_freq, bins)
        
        self.logger.info(f"PSD computed: {len(x)} frequency bins")
        
        # Generate plot
        # filename = kwargs.get("filename", "FITS File")
        fig = plot_psd_figure(x, y, title=f"Power Spectral Density - [{filename}]")
        
        # Save plot
        plot_id = str(uuid4())
        plot_bytes = self._fig_to_bytes(fig)
        FileManager.save_plot(plot_id, "psd", plot_bytes)
        plot_url = os.path.join(settings.plots_psd_dir, f"psd_{plot_id}.png")
        plt.close(fig)
        
        result = {
            "n_points": len(x),
            "freq_range": [float(x[0]), float(x[-1])],
            "psd_range": [float(np.min(y)), float(np.max(y))],
            "frequencies_sample": x.tolist()[:100],
            "psd_values_sample": y.tolist()[:100],
            "parameters_used": parameters
        }
        
        return (result, plot_url)
    
    def _fig_to_bytes(self, fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf.read()

# ==========================================
# Fitting Capability
# ==========================================

class FittingCapability(AnalysisCapability):
    """Capability for model fitting (power law, bending power law)"""
    
    def __init__(self, fitting_type: AnalysisType):
        self.fitting_type = fitting_type

        # Extract string value from Enum
        if isinstance(fitting_type, AnalysisType):
            self.fitting_type_str = fitting_type.value 
        else:
            self.fitting_type_str = str(fitting_type)
        
        self.logger = logging.getLogger(f"capability.{self.fitting_type_str}")
    
    def get_dependencies(self) -> list:
        """No dependencies - we compute PSD internally"""
        return []
    
    async def execute(
        self, 
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        file_record: Any
        # **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Fit power law model
        
        Parameters expected (with defaults from Classification Agent):
        Power Law: {
            "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
            "noise_bound_percent": 0.7,
            "A0": 1.0, "b0": 1.0,
            "A_min": 0.0, "A_max": 1e38,
            "b_min": 0.1, "b_max": 3.0,
            "maxfev": 1000000
        }
        
        Bending Power Law: {
            "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
            "noise_bound_percent": 0.7,
            "A0": 10.0, "fb0": 0.01, "sh0": 1.0,
            "A_min": 0.0, "A_max": 1e38,
            "fb_min": 2e-5, "fb_max": 0.05,
            "sh_min": 0.3, "sh_max": 3.0,
            "maxfev": 1000000
        }
        """

        low_freq = parameters.get("low_freq", 1e-5)
        high_freq = parameters.get("high_freq", 0.05)
        bins = parameters.get("bins", 3500)
        filename = parameters.get('filename', 'Unknown')

        try:
            freqs, psd = compute_psd(rate_data)
            
            # Validate frequency bounds (if function exists)
            try:
                from app.tools.psd import validate_frequency_bounds
                low_freq, high_freq = validate_frequency_bounds(
                    low_freq, high_freq, freqs
                )
            except ImportError:
                # If validate_frequency_bounds doesn't exist, do basic validation
                low_freq = max(low_freq, freqs.min())
                high_freq = min(high_freq, freqs.max())
            
            x, y = bin_psd(freqs, psd, low_freq, high_freq, bins)
            
            self.logger.info(
                f"PSD computed for fitting: {len(x)} frequency bins "
                f"in range [{low_freq:.6e}, {high_freq:.6e}] Hz"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compute PSD for fitting: {e}")
            raise ValueError(f"PSD computation failed: {str(e)}") from e
        
        # Fit model based on type
        if self.fitting_type == AnalysisType.POWER_LAW:
            return await self._fit_power_law(x, y, parameters, filename, file_record)
        elif self.fitting_type == AnalysisType.BENDING_POWER_LAW:
            return await self._fit_bending_power_law(x, y, parameters, filename, file_record)
        else:
            raise ValueError(f"Unknown fitting type: {self.fitting_type}")
    
    async def _fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parameters: dict,
        filename: str,
        file_record: Any
    ) -> tuple:
        """Fit power law model: PSD = A/f^b + n"""
        
        self.logger.info("Fitting power law model")
        
        # Extract parameters
        noise_bound_percent = parameters.get('noise_bound_percent', 0.7)
        A0 = parameters.get('A0', 1.0)
        b0 = parameters.get('b0', 1.0)
        A_min = parameters.get('A_min', 0.0)
        A_max = parameters.get('A_max', 1e38)
        b_min = parameters.get('b_min', 0.1)
        b_max = parameters.get('b_max', 3.0)
        maxfev = parameters.get('maxfev', 1000000)
        
        # Prepare initial parameters and bounds
        initial_params = {'A': A0, 'b': b0}
        param_bounds = {
            'A': (A_min, A_max),
            'b': (b_min, b_max)
        }
        
        try:
            # Fit model
            A, b, n = fit_power_law(
                x, y,
                noise_bound_percent=noise_bound_percent,
                initial_params=initial_params,
                param_bounds=param_bounds,
                maxfev=maxfev
            )
            
            self.logger.info(
                f"Power law fit completed: A={A:.6e}, b={b:.3f}, n={n:.6e}"
            )
            
            # Generate plot
            fig = plot_power_law_with_residual_figure(
                x, y, A, b, n,
                title=f"Power Law Fit - {filename}"
            )
            
            # Generate unique filename
            plot_id = str(uuid4())
            plot_filename = f"{self.fitting_type_str}_{plot_id}.png"  # ← ใช้ _str
            
            # Ensure directory exists
            os.makedirs(settings.plots_powerlaw_dir, exist_ok=True)
            
            # Build FULL file path (directory + filename)
            plot_path = os.path.join(settings.plots_powerlaw_dir, plot_filename)

            # Save figure to the FULL path
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Generate URL
            # plot_url = f"/storage/plots/{self.fitting_type_str}/{plot_filename}"  # ← ใช้ _str
            
            self.logger.info(f"Plot saved: {plot_path}")
            
            # Build result
            result = {
                "model": self.fitting_type_str,  # ← ใช้ _str
                "fitted_parameters": {
                    "A": float(A),
                    "b": float(b),
                    "n": float(n)
                },
                "initial_parameters": initial_params,
                "parameter_bounds": {
                    # k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
                    k: [
                        float(v[0]),
                        "unbounded" if np.isinf(float(v[1])) else float(v[1])
                    ]
                    for k, v in param_bounds.items()
                },
                "parameters_used": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in parameters.items()
                    if k != 'filename'
                }
            }
            
            return result, plot_path
            
        except Exception as e:
            self.logger.error(f"Power law fitting failed: {e}", exc_info=True)
            raise ValueError(f"Power law fitting failed: {str(e)}") from e
    
    async def _fit_bending_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
        parameters: dict,
        filename: str,
        file_record: Any
    ) -> tuple:
        """Fit bending power law model"""
        
        self.logger.info("Fitting bending power law model")
        
        # Extract parameters
        noise_bound_percent = parameters.get('noise_bound_percent', 0.7)
        A0 = parameters.get('A0', 10.0)
        fb0 = parameters.get('fb0', 0.01)
        sh0 = parameters.get('sh0', 1.0)
        A_min = parameters.get('A_min', 0.0)
        A_max = parameters.get('A_max', 1e38)
        fb_min = parameters.get('fb_min', 2e-5)
        fb_max = parameters.get('fb_max', 0.05)
        sh_min = parameters.get('sh_min', 0.3)
        sh_max = parameters.get('sh_max', 3.0)
        maxfev = parameters.get('maxfev', 1000000)
        
        # Prepare initial parameters
        initial_params = {'A': A0, 'sh': sh0}
        if fb0 is not None:
            initial_params['fb'] = fb0
        
        # Prepare bounds
        param_bounds = {
            'A': (A_min, A_max),
            'sh': (sh_min, sh_max)
        }
        if fb_min is not None or fb_max is not None:
            param_bounds['fb'] = (
                fb_min if fb_min is not None else x[0],
                fb_max if fb_max is not None else x[-1]
            )
        
        try:
            # Fit model
            A, fb, sh, n = fit_bending_power_law(
                x, y,
                noise_bound_percent=noise_bound_percent,
                initial_params=initial_params,
                param_bounds=param_bounds,
                maxfev=maxfev
            )
            
            self.logger.info(
                f"Bending power law fit completed: "
                f"A={A:.6e}, fb={fb:.6e}, sh={sh:.3f}, n={n:.6e}"
            )
            
            # Generate plot
            fig = plot_bending_power_law_with_residual_figure(
                x, y, A, fb, sh, n,
                title=f"Bending Power Law Fit - {filename}"
            )
            
            # Generate unique filename
            plot_id = str(uuid4())
            plot_filename = f"{self.fitting_type_str}_{plot_id}.png"  
           
            # Ensure directory exists
            os.makedirs(settings.plots_bendingpowerlaw_dir, exist_ok=True)

            # Build FULL file path (directory + filename)
            plot_path = os.path.join(settings.plots_bendingpowerlaw_dir, plot_filename)
            
            # Save figure to the FULL path
            fig.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Plot saved: {plot_path}")
            
            # Build result
            result = {
                "model": self.fitting_type_str, 
                "fitted_parameters": {
                    "A": float(A),
                    "fb": float(fb),
                    "sh": float(sh),
                    "n": float(n)
                },
                "initial_parameters": {
                    k: float(v) for k, v in initial_params.items()
                },
                "parameter_bounds": {
                    # k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
                    k: [
                        float(v[0]),
                        "unbounded" if np.isinf(float(v[1])) else float(v[1])
                    ]
                    for k, v in param_bounds.items()
                },
                "parameters_used": {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in parameters.items()
                    if k != 'filename'
                }
            }
            
            return result, plot_path
            
        except Exception as e:
            self.logger.error(f"Bending power law fitting failed: {e}", exc_info=True)
            raise ValueError(f"Bending power law fitting failed: {str(e)}") from e

# ==========================================
# Metadata Capability
# ==========================================

# class MetadataCapability(AnalysisCapability):
#     """Extract FITS file metadata"""
    
#     def __init__(self):
#         super().__init__("metadata")
    
#     async def execute(
#         self, 
#         rate_data: np.ndarray, 
#         parameters: Dict[str, Any],
#         **kwargs
#     ) -> Tuple[Dict[str, Any], Optional[str]]:
#         """
#         Extract metadata from file record
#         Note: rate_data not used for metadata extraction
#         """
#         file_record = kwargs.get("file_record")
        
#         if not file_record:
#             raise ValueError("file_record required for metadata extraction")
        
#         self.logger.info(f"Extracting metadata from file: {file_record.file_id}")
        
#         result = {
#             "file_id": str(file_record.file_id),
#             "original_filename": file_record.original_filename,
#             "metadata_filename": file_record.metadata_filename,
#             "file_size": file_record.file_size,
#             "uploaded_at": file_record.uploaded_at.isoformat(),
#             "is_valid": file_record.is_valid,
#             "validation_status": file_record.validation_status,
#             "fits_metadata": file_record.fits_metadata,
#             "data_info": file_record.data_info
#         }
        
#         self.logger.info("Metadata extraction completed")
        
#         return (result, None)  # No plot for metadataa

class MetadataCapability(AnalysisCapability):
    """
    Extract FITS file metadata with enhanced analysis context
    
    Provides three levels of metadata:
    1. critical_fields: Essential observation parameters
    2. derived_quantities: Computed values for analysis interpretation
    3. source_context: Known source information from literature
    """
    
    # Database of known sources
    KNOWN_SOURCES = {
        'IRAS 13224-3809': {
            'type': 'Narrow-Line Seyfert 1 (NLS1) AGN',
            'redshift': 0.06576,
            'typical_luminosity': '~10^44 erg/s',
            'black_hole_mass_solar': 2e6,
            'characteristics': [
                'Extreme X-ray variability',
                'Soft X-ray excess',
                'Relativistic reflection features'
            ],
            'typical_psd': {
                'power_law_index_range': [1.2, 1.8],
                'break_frequency_range': [1e-4, 1e-3],
                'notes': 'Often shows bending power law with break around few × 10⁻⁴ Hz'
            },
            'key_references': [
                'Alston et al. (2020) MNRAS 482, 2088',
                'Chiang et al. (2015) MNRAS 446, 759',
                'Fabian et al. (2013) MNRAS 429, 2917'
            ]
        },
        'Cyg X-1': {
            'type': 'Black hole X-ray binary',
            'black_hole_mass_solar': 21.2,
            'characteristics': [
                'Hard and soft spectral states',
                'Strong QPOs in hard state',
                'Persistent source'
            ],
            'typical_psd': {
                'power_law_index_range': [0.7, 1.5],
                'break_frequency_range': [0.01, 1.0],
                'notes': 'State-dependent PSD shape'
            },
            'key_references': [
                'Nowak et al. (1999) ApJ 510, 874'
            ]
        },
        'GRS 1915+105': {
            'type': 'Black hole X-ray binary',
            'black_hole_mass_solar': 12.4,
            'characteristics': [
                'Multiple variability classes',
                'Superluminal jets',
                'Extreme luminosity'
            ],
            'typical_psd': {
                'power_law_index_range': [0.5, 1.5],
                'break_frequency_range': [0.001, 0.1],
                'notes': 'Highly variable PSD depending on state'
            }
        }
    }
    
    def __init__(self):
        super().__init__("metadata")

    
    async def execute(
        self, 
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Extract metadata WITH analysis-relevant computed fields
        
        Returns:
            Tuple of (result_dict, None)
            result_dict contains:
            - Basic metadata (file info)
            - critical_fields (essential parameters)
            - derived_quantities (computed values)
            - source_context (literature info)
        """
        file_record = kwargs.get("file_record")
        
        if not file_record:
            raise ValueError("file_record required for metadata extraction")
        
        self.logger.info(f"Extracting enhanced metadata from file: {file_record.file_id}")
            
        # ============================================
        # STEP 1: Basic metadata (existing)
        # ============================================
        basic_metadata = {
            "file_id": str(file_record.file_id),
            "original_filename": file_record.original_filename,
            "metadata_filename": file_record.metadata_filename,
            "file_size": file_record.file_size,
            "uploaded_at": file_record.uploaded_at.isoformat(),
            "is_valid": file_record.is_valid,
            "validation_status": file_record.validation_status,
        }

        # ============================================
        # STEP 2: Extract FITS header
        # ============================================
        fits_metadata = file_record.fits_metadata or {}
        data_info = file_record.data_info or {}
        
        # ============================================
        # STEP 3: Extract critical fields
        # ============================================
        critical_fields = self._extract_critical_fields(fits_metadata, data_info)
        
        # ============================================
        # STEP 4: Compute derived quantities
        # ============================================
        derived_quantities = self._compute_derived_quantities(
            fits_metadata, 
            data_info, 
            rate_data
        )
        
        # ============================================
        # STEP 5: Get source context
        # ============================================
        source_context = self._get_source_context(fits_metadata)
        
        # ============================================
        # STEP 6: Combine everything
        # ============================================
        result = {
            **basic_metadata,
            
            # Enhanced fields for AstroSage
            "critical_fields": critical_fields,
            "derived_quantities": derived_quantities,
            "source_context": source_context,
            
            # Keep original for reference (optional)
            "fits_metadata": fits_metadata,
            "data_info": data_info
        }
        
        self.logger.info("Enhanced metadata extraction completed")
        
        return (result, None)
    
    def _extract_critical_fields(
        self, 
        fits_metadata: Dict, 
        data_info: Dict
    ) -> Dict[str, Any]:
        """
        Extract only the most important fields
        
        Returns:
            Dictionary with critical observation parameters
        """
        critical = {}
        
        # Source identification
        critical['source_name'] = fits_metadata.get('OBJECT')
        critical['telescope'] = fits_metadata.get('TELESCOP')
        critical['instrument'] = fits_metadata.get('INSTRUME')
        critical['observation_id'] = fits_metadata.get('OBS_ID')
        critical['observer'] = fits_metadata.get('OBSERVER')
        
        # Extract from HDU 1 (RATE table)
        rate_hdu = self._find_rate_hdu(data_info)
        
        if rate_hdu:
            header_dict = {
                card['keyword']: card['value']
                for card in rate_hdu.get('header_cards', [])
            }
            
            # Timing parameters
            critical['time_bin_size'] = float(header_dict.get('TIMEDEL', 0))
            critical['t_start'] = float(header_dict.get('TSTART', 0))
            critical['t_stop'] = float(header_dict.get('TSTOP', 0))
            critical['exposure_time'] = float(header_dict.get('EXPOSURE', 0))
            critical['good_time'] = float(header_dict.get('ONTIME', 0))
            
            # Energy parameters
            critical['energy_min_ev'] = float(header_dict.get('CHANMIN', 0))
            critical['filter_mode'] = header_dict.get('FILTER')
            
            # Data quality indicators
            critical['background_subtracted'] = header_dict.get('BACKAPP') == 'True'
            critical['dead_time_corrected'] = header_dict.get('DEADAPP') == 'True'
            critical['vignetting_corrected'] = header_dict.get('VIGNAPP') == 'True'
            critical['background_ratio'] = float(header_dict.get('BKGRATIO', 0))
            
            # Data structure
            critical['n_time_bins'] = rate_hdu.get('n_rows', 0)
            critical['has_error_column'] = 'ERROR' in rate_hdu.get('column_names', [])
        
        # GTI information
        gti_hdu = self._find_gti_hdu(data_info)
        if gti_hdu:
            critical['n_gti_segments'] = gti_hdu.get('n_rows', 0)
        
        return critical
    
    def _compute_derived_quantities(
        self,
        fits_metadata: Dict,
        data_info: Dict,
        rate_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute useful derived quantities
        
        Returns:
            Dictionary with computed analysis-relevant values
        """
        derived = {}
        
        rate_hdu = self._find_rate_hdu(data_info)
        
        if not rate_hdu:
            return derived
        
        header_dict = {
            card['keyword']: card['value']
            for card in rate_hdu.get('header_cards', [])
        }
        
        # Extract basic parameters
        timedel = float(header_dict.get('TIMEDEL', 1.0))
        tstart = float(header_dict.get('TSTART', 0))
        tstop = float(header_dict.get('TSTOP', 0))
        exposure = float(header_dict.get('EXPOSURE', 0))
        ontime = float(header_dict.get('ONTIME', 0))
        
        # ============================================
        # Frequency constraints
        # ============================================
        if timedel > 0:
            derived['nyquist_frequency_hz'] = 1.0 / (2 * timedel)
        
        duration = tstop - tstart
        if duration > 0:
            derived['observation_duration_seconds'] = duration
            derived['observation_duration_hours'] = duration / 3600
            derived['min_frequency_hz'] = 1.0 / duration
            derived['duty_cycle_percent'] = (ontime / duration) * 100
        
        # Recommended frequency range (conservative)
        if 'min_frequency_hz' in derived and 'nyquist_frequency_hz' in derived:
            f_min = derived['min_frequency_hz']
            f_nyq = derived['nyquist_frequency_hz']
            
            # Avoid 1/T effects at low freq, Nyquist artifacts at high freq
            derived['recommended_f_low_hz'] = 5 * f_min
            derived['recommended_f_high_hz'] = 0.1 * f_nyq
        
        # ============================================
        # Energy band classification
        # ============================================
        chanmin = float(header_dict.get('CHANMIN', 0))
        if chanmin > 0:
            chanmin_kev = chanmin / 1000  # eV to keV
            derived['energy_min_kev'] = chanmin_kev
            
            if chanmin_kev < 2:
                derived['energy_band'] = 'soft X-ray (< 2 keV)'
                derived['energy_band_short'] = 'soft X-ray'
            elif chanmin_kev < 10:
                derived['energy_band'] = 'hard X-ray (2-10 keV)'
                derived['energy_band_short'] = 'hard X-ray'
            else:
                derived['energy_band'] = 'very hard X-ray (> 10 keV)'
                derived['energy_band_short'] = 'very hard X-ray'
        
        # ============================================
        # Data rate statistics
        # ============================================
        # if rate_data is not None and rate_data.size > 0:
        #     derived['mean_count_rate'] = float(np.mean(rate_data))
        #     derived['median_count_rate'] = float(np.median(rate_data))
        #     derived['std_count_rate'] = float(np.std(rate_data))
            
        #     if derived['mean_count_rate'] > 0:
        #         derived['count_rate_variability'] = float(
        #             derived['std_count_rate'] / derived['mean_count_rate']
        #         )
                
        #         # Estimate Poisson noise level (for rms-normalized PSD)
        #         derived['expected_poisson_noise_rms'] = 2.0 / derived['mean_count_rate']
        
        # ============================================
        # Exposure efficiency
        # ============================================
        if exposure > 0 and duration > 0:
            derived['exposure_efficiency_percent'] = (exposure / duration) * 100
        
        return derived
    
    def _get_source_context(self, fits_metadata: Dict) -> Dict[str, Any]:
        """
        Get context about the source from known database
        
        Returns:
            Dictionary with source context from literature
        """
        source_name = fits_metadata.get('OBJECT', '').strip()
        
        context = {
            'is_known_source': source_name in self.KNOWN_SOURCES,
            'source_name': source_name
        }
        
        if source_name in self.KNOWN_SOURCES:
            # Copy all known information
            source_info = self.KNOWN_SOURCES[source_name].copy()
            context.update(source_info)
            
            self.logger.info(f"Found known source: {source_name} ({source_info.get('type')})")
        else:
            self.logger.info(f"Unknown source: {source_name}")
        
        return context
    
    def _find_rate_hdu(self, data_info: Dict) -> Optional[Dict]:
        """Find RATE table HDU (usually HDU 1)"""
        for hdu in data_info.get('hdu_info', []):
            if hdu.get('hdu_index') == 1:
                return hdu
            # Also check if RATE is in column names
            if 'RATE' in hdu.get('column_names', []):
                return hdu
        return None
    
    def _find_gti_hdu(self, data_info: Dict) -> Optional[Dict]:
        """Find GTI table HDU"""
        for hdu in data_info.get('hdu_info', []):
            cards = hdu.get('header_cards', [])
            for card in cards:
                if card.get('keyword') == 'EXTNAME':
                    extname = str(card.get('value', ''))
                    if 'GTI' in extname or 'SRC_GTIS' in extname:
                        return hdu
        return None