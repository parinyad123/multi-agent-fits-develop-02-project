
"""
multi-agent-fits-dev-02/app/agents/analysis/capabilities/implementations.py

Concrete implementations of analysis capabilities
"""

from typing import Dict, Any, Optional, Tuple
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
import io

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
        # self.logger.info(f"Executing statistics analysis with {len(rate_data)} data points, {type(rate_data)}")
        # ========================================
        # STEP 1: CRITICAL TYPE VALIDATION
        # ========================================
        self.logger.info(f"Starting statistics analysis")
        self.logger.debug(f"Received rate_data type: {type(rate_data)}")
        self.logger.debug(f"Received rate_data value: {rate_data}")

        # Check if rate_data is None
        if rate_data is None:
            error_msg = "rate_data is None"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check if rate_data is numpy array
        if not isinstance(rate_data, np.ndarray):
            error_msg = (
                f"rate_data must be numpy.ndarray, got {type(rate_data)}. "
                f"Value: {rate_data}"
            )
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Check if array is empty
        if rate_data.size == 0:  # ← ใช้ .size แทน len() เพื่อความปลอดภัย
            error_msg = f"rate_data is empty (size: {rate_data.size})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check array shape
        if rate_data.ndim != 1:
            error_msg = (
                f"rate_data must be 1-dimensional, got {rate_data.ndim} dimensions. "
                f"Shape: {rate_data.shape}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log successful validation
        self.logger.info(
            f"✓ Validation passed: "
            f"type={type(rate_data).__name__}, "
            f"shape={rate_data.shape}, "
            f"size={rate_data.size}, "
            f"dtype={rate_data.dtype}"
        )
        
        # ========================================
        # STEP 2: EXTRACT PARAMETERS
        # ========================================
        metrics = parameters.get("metrics", ["mean", "median", "std", "min", "max", "count"])
        percentiles = parameters.get("percentiles", [])
        quantiles = parameters.get("quantiles", [])

        self.logger.info(
            f"Executing statistics: "
            f"{len(metrics)} metrics, "
            f"{len(percentiles)} percentiles, "
            f"{len(quantiles)} quantiles"
        )

        # # validation
        # if rate_data is None:
        #     raise ValueError("rate_data cannot be None")
        
        # if len(rate_data) == 0:
        #     raise ValueError("rate_data is empty")

        # Calculate basic statistics
        # stats = calculate_statistics(rate_data, metrics)

        # ========================================
        # STEP 3: CALCULATE BASIC STATISTICS
        # ========================================
        try:
            stats = calculate_statistics(rate_data, metrics)
            self.logger.debug(f"Basic statistics calculated: {list(stats.keys())}")
        except Exception as e:
            error_msg = f"Failed to calculate basic statistics: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        
        # ========================================
        # STEP 4: CALCULATE PERCENTILES
        # ========================================
        if percentiles:
            self.logger.info(f"Computing percentiles: {percentiles}")
            percentile_results = {}
            
            try:
                for p in percentiles:
                    if not (0 <= p <= 100):
                        self.logger.warning(f"Invalid percentile {p}, must be 0-100. Skipping.")
                        continue
                    
                    value = float(np.percentile(rate_data, p))
                    percentile_results[f"percentile_{p}"] = value
                
                stats.update(percentile_results)
                self.logger.info(f"Percentiles computed: {list(percentile_results.keys())}")
                
            except Exception as e:
                self.logger.error(f"Error computing percentiles: {e}", exc_info=True)
                # Don't fail entirely, continue with what we have

        # ========================================
        # STEP 5: CALCULATE QUANTILES
        # ========================================
        if quantiles:
            self.logger.info(f"Computing quantiles: {quantiles}")
            quantile_results = {}
            
            try:
                for q in quantiles:
                    if not (0 <= q <= 1):
                        self.logger.warning(f"Invalid quantile {q}, must be 0-1. Skipping.")
                        continue
                    
                    value = float(np.quantile(rate_data, q))
                    # Format quantile key (e.g., 0.25 → quantile_0_25)
                    q_key = f"quantile_{str(q).replace('.', '_')}"
                    quantile_results[q_key] = value
                
                stats.update(quantile_results)
                self.logger.info(f"Quantiles computed: {list(quantile_results.keys())}")
                
            except Exception as e:
                self.logger.error(f"Error computing quantiles: {e}", exc_info=True)
                # Don't fail entirely, continue with what we have
        
        # ========================================
        # STEP 6: ADD DISTRIBUTION SUMMARY
        # ========================================
        if percentiles or quantiles:
            try:
                stats["distribution_summary"] = self._create_distribution_summary(
                    rate_data, stats
                )
                self.logger.debug("Distribution summary created")
            except Exception as e:
                self.logger.warning(f"Could not create distribution summary: {e}")
                # Optional feature, don't fail
        
        # ========================================
        # STEP 7: PREPARE RESULT
        # ========================================
        result = {
            "statistics": stats,
            "n_data_points": int(rate_data.size),  # ← ใช้ .size แทน len()
            "data_info": {
                "dtype": str(rate_data.dtype),
                "shape": list(rate_data.shape),
                "memory_size_bytes": int(rate_data.nbytes)
            },
            "parameters_used": parameters
        }
        
        self.logger.info(
            f"Statistics completed successfully: "
            f"{len(stats)} metrics computed "
            f"(including {len(percentiles)} percentiles, {len(quantiles)} quantiles)"
        )
        
        return (result, None)  # No plot for statistics
    
    def _create_distribution_summary(
        self, 
        rate_data: np.ndarray, 
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create distribution summary from computed statistics
        
        Returns summary including:
        - Range (min to max)
        - IQR (Interquartile Range)
        - Spread metrics
        """
        summary = {}
        
        # Data range
        if "min" in stats and "max" in stats:
            summary["range"] = {
                "min": stats["min"],
                "max": stats["max"],
                "span": stats["max"] - stats["min"]
            }
        
        # IQR (Interquartile Range) - if we have 25th and 75th percentiles
        q25_key = None
        q75_key = None
        
        # Check for percentile format
        if "percentile_25" in stats:
            q25_key = "percentile_25"
        # Check for quantile format
        elif "quantile_0_25" in stats:
            q25_key = "quantile_0_25"
        
        if "percentile_75" in stats:
            q75_key = "percentile_75"
        elif "quantile_0_75" in stats:
            q75_key = "quantile_0_75"
        
        if q25_key and q75_key:
            q25 = stats[q25_key]
            q75 = stats[q75_key]
            iqr = q75 - q25
            
            summary["iqr"] = {
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "lower_fence": q25 - 1.5 * iqr,  # Outlier detection
                "upper_fence": q75 + 1.5 * iqr
            }
        
        # Coefficient of variation (if mean and std available)
        if "mean" in stats and "std" in stats and stats["mean"] != 0:
            summary["coefficient_of_variation"] = stats["std"] / abs(stats["mean"])
        
        # Skewness and Kurtosis
        try:
            summary["skewness"] = float(scipy_stats.skew(rate_data))
            summary["kurtosis"] = float(scipy_stats.kurtosis(rate_data))
        except Exception as e:
            self.logger.warning(f"Could not calculate skewness/kurtosis: {e}")
        
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
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Compute PSD and generate plot
        
        Parameters expected (with defaults from Classification Agent):
        {
            "low_freq": 1e-5,
            "high_freq": 0.05,
            "bins": 3500
        }
        """
        # ========================================
        # ADD VALIDATION (same as StatisticsCapability)
        # ========================================
        self.logger.info("Starting PSD computation")
        self.logger.debug(f"Received rate_data type: {type(rate_data)}")
        
        if rate_data is None:
            raise ValueError("rate_data is None")
        
        if not isinstance(rate_data, np.ndarray):
            raise TypeError(f"rate_data must be numpy.ndarray, got {type(rate_data)}")
        
        if rate_data.size == 0:
            raise ValueError(f"rate_data is empty")
        
        self.logger.info(f"✓ Validation passed: shape={rate_data.shape}, size={rate_data.size}")
        
        # Extract parameters
        low_freq = parameters.get("low_freq", 1e-5)
        high_freq = parameters.get("high_freq", 0.05)
        bins = parameters.get("bins", 3500)
        
        # Compute PSD
        freqs, psd = compute_psd(rate_data)
        
        # Bin PSD
        x, y = bin_psd(freqs, psd, low_freq, high_freq, bins)
        
        self.logger.info(f"PSD computed: {len(x)} frequency bins")
        
        # Generate plot
        filename = kwargs.get("filename", "FITS File")
        fig = plot_psd_figure(x, y, title=f"Power Spectral Density - [{filename}]")
        
        # Save plot
        plot_id = str(uuid4())
        plot_bytes = self._fig_to_bytes(fig)
        plot_path = FileManager.save_plot(plot_id, "psd", plot_bytes)
        plot_url = f"/storage/plots/psd/psd_{plot_id}.png"
        
        plt.close(fig)
        
        self.logger.info(f"PSD plot saved: {plot_url}")
        
        # Prepare result (sample data to reduce response size)
        result = {
            "n_points": len(x),
            "freq_range": [float(x[0]), float(x[-1])],
            "psd_range": [float(np.min(y)), float(np.max(y))],
            "frequencies_sample": x.tolist()[:100],  # First 100 points
            "psd_values_sample": y.tolist()[:100],
            "parameters_used": parameters
        }
        
        return (result, plot_url)
    
    def _fig_to_bytes(self, fig) -> bytes:
        """Convert matplotlib figure to PNG bytes"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf.read()

# ==========================================
# Fitting Capability
# ==========================================

class FittingCapability(AnalysisCapability):
    """Fit power law models to PSD data"""
    
    def __init__(self, model_type: str):
        """
        Initialize fitting capability
        
        Args:
            model_type: "power_law" or "bending_power_law"
        """
        super().__init__(f"fitting_{model_type}")
        self.model_type = model_type
    
    def get_dependencies(self) -> list:
        """Fitting requires PSD to be computed first"""
        return ["psd"]
    
    async def execute(
        self, 
        rate_data: np.ndarray, 
        parameters: Dict[str, Any],
        **kwargs
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
        self.logger.info(f"Starting {self.model_type} fitting")
        self.logger.debug(f"Received rate_data type: {type(rate_data)}")
        
        if rate_data is None:
            raise ValueError("rate_data is None")
        
        if not isinstance(rate_data, np.ndarray):
            raise TypeError(f"rate_data must be numpy.ndarray, got {type(rate_data)}")
        
        if rate_data.size == 0:
            raise ValueError(f"rate_data is empty")
        
        self.logger.info(f"✓ Validation passed: shape={rate_data.shape}, size={rate_data.size}")
        
        # Compute PSD first (needed for fitting)
        low_freq = parameters.get("low_freq", 1e-5)
        high_freq = parameters.get("high_freq", 0.05)
        bins = parameters.get("bins", 3500)
        
        freqs, psd = compute_psd(rate_data)
        x, y = bin_psd(freqs, psd, low_freq, high_freq, bins)
        
        # Fit model
        if self.model_type == "power_law":
            fitted_params, fig = await self._fit_power_law(x, y, parameters)
        else:  # bending_power_law
            fitted_params, fig = await self._fit_bending_power_law(x, y, parameters)
        
        # Save plot
        plot_id = str(uuid4())
        plot_bytes = self._fig_to_bytes(fig)
        plot_path = FileManager.save_plot(plot_id, self.model_type, plot_bytes)
        plot_url = f"/storage/plots/{self.model_type}/{self.model_type}_{plot_id}.png"
        
        plt.close(fig)
        
        self.logger.info(f"{self.model_type} fitting completed: {plot_url}")
        
        result = {
            "model": self.model_type,
            "fitted_parameters": fitted_params,
            "parameters_used": parameters
        }
        
        return (result, plot_url)
    
    async def _fit_power_law(self, x, y, parameters):
        """Fit power law model"""
        # Extract fitting parameters
        noise_bound_percent = parameters.get("noise_bound_percent", 0.7)
        A0 = parameters.get("A0", 1.0)
        b0 = parameters.get("b0", 1.0)
        A_min = parameters.get("A_min", 0.0)
        A_max = parameters.get("A_max", 1e38)
        b_min = parameters.get("b_min", 0.1)
        b_max = parameters.get("b_max", 3.0)
        maxfev = parameters.get("maxfev", 1000000)
        
        # Fit
        A, b, n = fit_power_law(
            x, y,
            noise_bound_percent=noise_bound_percent,
            initial_params={"A": A0, "b": b0},
            param_bounds={
                "A": (A_min, A_max),
                "b": (b_min, b_max)
            },
            maxfev=maxfev
        )
        
        # Generate plot
        filename = parameters.get("filename", "FITS File")
        fig = plot_power_law_with_residual_figure(
            x, y, A, b, n, 
            title=f"Power Law Fit - [{filename}]"
        )
        
        fitted_params = {
            "A": float(A),
            "b": float(b),
            "n": float(n)
        }
        
        return fitted_params, fig
    
    async def _fit_bending_power_law(self, x, y, parameters):
        """Fit bending power law model"""
        # Extract fitting parameters
        noise_bound_percent = parameters.get("noise_bound_percent", 0.7)
        A0 = parameters.get("A0", 10.0)
        fb0 = parameters.get("fb0", 0.01)  
        sh0 = parameters.get("sh0", 1.0)
        A_min = parameters.get("A_min", 0.0)
        A_max = parameters.get("A_max", 1e38)
        fb_min = parameters.get("fb_min", 2e-5)
        fb_max = parameters.get("fb_max", 0.05)
        sh_min = parameters.get("sh_min", 0.3)
        sh_max = parameters.get("sh_max", 3.0)
        maxfev = parameters.get("maxfev", 1000000)
        
        # Build initial params
        initial_params = {"A": A0, "sh": sh0}
        if fb0 is not None:
            initial_params["fb"] = fb0
        
        # Build param bounds
        param_bounds = {
            "A": (A_min, A_max),
            "sh": (sh_min, sh_max)
        }
        if fb_min is not None or fb_max is not None:
            param_bounds["fb"] = (
                fb_min if fb_min is not None else x[0],
                fb_max if fb_max is not None else x[-1]
            )
        
        # Fit
        A, fb, sh, n = fit_bending_power_law(
            x, y,
            noise_bound_percent=noise_bound_percent,
            initial_params=initial_params,
            param_bounds=param_bounds,
            maxfev=maxfev
        )
        
        # Generate plot
        filename = parameters.get("filename", "FITS File")
        fig = plot_bending_power_law_with_residual_figure(
            x, y, A, fb, sh, n,
            title=f"Bending Power Law Fit - [{filename}]"
        )
        
        fitted_params = {
            "A": float(A),
            "fb": float(fb),
            "sh": float(sh),
            "n": float(n)
        }
        
        return fitted_params, fig
    
    def _fig_to_bytes(self, fig) -> bytes:
        """Convert matplotlib figure to PNG bytes"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return buf.read()

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