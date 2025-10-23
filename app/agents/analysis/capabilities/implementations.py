
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
        # ========================================
        # ADD VALIDATION
        # ========================================
        # self.logger.info(f"Starting {self.model_type} fitting")
        # self.logger.debug(f"Received rate_data type: {type(rate_data)}")
        
        # if rate_data is None:
        #     raise ValueError("rate_data is None")
        
        # if not isinstance(rate_data, np.ndarray):
        #     raise TypeError(f"rate_data must be numpy.ndarray, got {type(rate_data)}")
        
        # if rate_data.size == 0:
        #     raise ValueError(f"rate_data is empty")
        
        # self.logger.info(f"✓ Validation passed: shape={rate_data.shape}, size={rate_data.size}")
        
        # Compute PSD first (needed for fitting)
        low_freq = parameters.get("low_freq", 1e-5)
        high_freq = parameters.get("high_freq", 0.05)
        bins = parameters.get("bins", 3500)
        filename = parameters.get('filename', 'Unknown')
        
        # Compute PSD internally
        # self.logger.debug("Computing PSD for fitting")
        
        # freqs, psd = compute_psd(rate_data)
        # x, y = bin_psd(freqs, psd, low_freq, high_freq, bins)
        
        # # Fit model
        # if self.model_type == "power_law":
        #     fitted_params, fig = await self._fit_power_law(x, y, parameters)
        # else:  # bending_power_law
        #     fitted_params, fig = await self._fit_bending_power_law(x, y, parameters)
        
        # # Save plot
        # plot_id = str(uuid4())
        # plot_bytes = self._fig_to_bytes(fig)
        # plot_path = FileManager.save_plot(plot_id, self.model_type, plot_bytes)
        # plot_url = f"/storage/plots/{self.model_type}/{self.model_type}_{plot_id}.png"
        
        # plt.close(fig)
        
        # self.logger.info(f"{self.model_type} fitting completed: {plot_url}")
        
        # result = {
        #     "model": self.model_type,
        #     "fitted_parameters": fitted_params,
        #     "parameters_used": parameters
        # }
        
        # return (result, plot_url)

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
    
    # async def _fit_power_law(self, x, y, parameters):
    #     """Fit power law model"""
    #     # Extract fitting parameters
    #     noise_bound_percent = parameters.get("noise_bound_percent", 0.7)
    #     A0 = parameters.get("A0", 1.0)
    #     b0 = parameters.get("b0", 1.0)
    #     A_min = parameters.get("A_min", 0.0)
    #     A_max = parameters.get("A_max", 1e38)
    #     b_min = parameters.get("b_min", 0.1)
    #     b_max = parameters.get("b_max", 3.0)
    #     maxfev = parameters.get("maxfev", 1000000)
        
    #     # Fit
    #     A, b, n = fit_power_law(
    #         x, y,
    #         noise_bound_percent=noise_bound_percent,
    #         initial_params={"A": A0, "b": b0},
    #         param_bounds={
    #             "A": (A_min, A_max),
    #             "b": (b_min, b_max)
    #         },
    #         maxfev=maxfev
    #     )
        
    #     # Generate plot
    #     filename = parameters.get("filename", "FITS File")
    #     fig = plot_power_law_with_residual_figure(
    #         x, y, A, b, n, 
    #         title=f"Power Law Fit - [{filename}]"
    #     )
        
    #     fitted_params = {
    #         "A": float(A),
    #         "b": float(b),
    #         "n": float(n)
    #     }
        
    #     return fitted_params, fig
    
    # async def _fit_bending_power_law(self, x, y, parameters):
    #     """Fit bending power law model"""
    #     # Extract fitting parameters
    #     noise_bound_percent = parameters.get("noise_bound_percent", 0.7)
    #     A0 = parameters.get("A0", 10.0)
    #     fb0 = parameters.get("fb0", 0.01)  
    #     sh0 = parameters.get("sh0", 1.0)
    #     A_min = parameters.get("A_min", 0.0)
    #     A_max = parameters.get("A_max", 1e38)
    #     fb_min = parameters.get("fb_min", 2e-5)
    #     fb_max = parameters.get("fb_max", 0.05)
    #     sh_min = parameters.get("sh_min", 0.3)
    #     sh_max = parameters.get("sh_max", 3.0)
    #     maxfev = parameters.get("maxfev", 1000000)
        
    #     # Build initial params
    #     initial_params = {"A": A0, "sh": sh0}
    #     if fb0 is not None:
    #         initial_params["fb"] = fb0
        
    #     # Build param bounds
    #     param_bounds = {
    #         "A": (A_min, A_max),
    #         "sh": (sh_min, sh_max)
    #     }
    #     if fb_min is not None or fb_max is not None:
    #         param_bounds["fb"] = (
    #             fb_min if fb_min is not None else x[0],
    #             fb_max if fb_max is not None else x[-1]
    #         )
        
    #     # Fit
    #     A, fb, sh, n = fit_bending_power_law(
    #         x, y,
    #         noise_bound_percent=noise_bound_percent,
    #         initial_params=initial_params,
    #         param_bounds=param_bounds,
    #         maxfev=maxfev
    #     )
        
    #     # Generate plot
    #     filename = parameters.get("filename", "FITS File")
    #     fig = plot_bending_power_law_with_residual_figure(
    #         x, y, A, fb, sh, n,
    #         title=f"Bending Power Law Fit - [{filename}]"
    #     )
        
    #     fitted_params = {
    #         "A": float(A),
    #         "fb": float(fb),
    #         "sh": float(sh),
    #         "n": float(n)
    #     }
        
    #     return fitted_params, fig
    
    # def _fig_to_bytes(self, fig) -> bytes:
    #     """Convert matplotlib figure to PNG bytes"""
    #     buf = io.BytesIO()
    #     fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    #     buf.seek(0)
    #     return buf.read()
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
                    k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
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
                    k: [float(v[0]), "unbounded" if np.isinf(v[1]) else float(v[1])]
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

class MetadataCapability(AnalysisCapability):
    """Extract FITS file metadata"""
    
    def __init__(self):
        super().__init__("metadata")
    
    async def execute(
        self, 
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Extract metadata from file record
        Note: rate_data not used for metadata extraction
        """
        file_record = kwargs.get("file_record")
        
        if not file_record:
            raise ValueError("file_record required for metadata extraction")
        
        self.logger.info(f"Extracting metadata from file: {file_record.file_id}")
        
        result = {
            "file_id": str(file_record.file_id),
            "original_filename": file_record.original_filename,
            "metadata_filename": file_record.metadata_filename,
            "file_size": file_record.file_size,
            "uploaded_at": file_record.uploaded_at.isoformat(),
            "is_valid": file_record.is_valid,
            "validation_status": file_record.validation_status,
            "fits_metadata": file_record.fits_metadata,
            "data_info": file_record.data_info
        }
        
        self.logger.info("Metadata extraction completed")
        
        return (result, None)  # No plot for metadataa