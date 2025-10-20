"""
multi-agent-fits-dev-02/app/tools/fitting.py

Model fitting utilities for Power Spectral Density (PSD) analysis
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ==========================================
# Model Functions
# ==========================================

def power_law_fn(f: np.ndarray, A: float, b: float, n: float) -> np.ndarray:
    """
    Power law model function: PSD(f) = A/f^b + n
    
    This function represents a simple power law model with a frequency-independent
    noise level. Power laws are common in many astrophysical processes, including
    accretion onto black holes and other compact objects.
    
    Args:
        f: Frequency array
        A: Amplitude
        b: Power law index
        n: Frequency-independent noise level
    
    Returns:
        Power spectral density values according to the model
    """
    return (A / (f ** b)) + n

def bending_power_law_fn(
    f: np.ndarray, 
    A: float, 
    fb: float, 
    sh: float, 
    n: float
) -> np.ndarray:
    """
    Bending power law model function: PSD(f) = A / [f(1+(f/fb)^(sh-1))] + n
    
    This function represents a bending power law model, which shows different
    power law slopes at low and high frequencies, with a smooth transition at
    the break frequency (fb). This model is particularly useful for systems
    with characteristic timescales, such as accretion disks with inner truncation radii.
    
    Args:
        f: Frequency array
        A: Amplitude
        fb: Break frequency
        sh: Shape parameter controlling the sharpness of the transition
        n: Frequency-independent noise level
    
    Returns:
        Power spectral density values according to the model
    """
    return (A / (f * (1 + (f / fb) ** (sh - 1)))) + n

# ==========================================
# Fitting Functions
# ==========================================

def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    noise_bound_percent: float = 0.7,
    maxfev: int = 1000000,
    initial_params: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[float, float, float]:
    """
    Fit a power law model to frequency data
    
    This function fits the model PSD(f) = A/f^b + n to provided
    frequency and PSD data, using a two-step approach with curve_fit from scipy.optimize.
    This improves parameter estimation accuracy, especially for the noise level.
    
    Args:
        x: Frequency array
        y: Power spectral density array
        noise_bound_percent: Controls the allowed range for the noise parameter 
                           relative to the initial estimate (default: 0.7)
        maxfev: Maximum number of function evaluations (default: 1000000)
        initial_params: Optional dictionary with custom initial parameter guesses:
                - 'A': Amplitude
                - 'b': Power law index
        param_bounds: Optional dictionary with custom parameter bounds:
                - 'A': (min, max) for amplitude
                - 'b': (min, max) for power law index
    
    Returns:
        Tuple of fitted parameters: (A, b, n):
            - A: Amplitude
            - b: Power law index
            - n: Frequency-independent noise level
    
    Example:
        >>> x = np.logspace(-5, -1, 100)
        >>> y = 1.0 / x**1.5 + 0.01
        >>> A, b, n = fit_power_law(x, y)
        >>> print(f"A={A:.3e}, b={b:.2f}, n={n:.3e}")
    """
    
    try:
        # Default values
        initial_params = initial_params or {}
        param_bounds = param_bounds or {}
        
        # Estimate the noise level from the high-frequency end of the spectrum
        # This is always calculated from the data and not customizable through initial_params
        n0 = float(np.mean(y[-10:]))
        
        # Initial guesses for parameters
        A0 = initial_params.get('A', 1.0)
        b0 = initial_params.get('b', 1.0)
        
        p0 = [A0, b0, n0]
        
        # Set bounds for the noise parameter based on noise_bound_percent
        n_bound_min = n0 * (1 - noise_bound_percent)
        n_bound_max = n0 * (1 + noise_bound_percent)
        
        # Set bounds for all parameters
        A_bounds = param_bounds.get('A', (0, 15))
        b_bounds = param_bounds.get('b', (0.1, 3.0))
        
        # Handle infinity in bounds
        A_min = A_bounds[0] if not np.isinf(A_bounds[0]) else 0.0
        A_max = A_bounds[1] if not np.isinf(A_bounds[1]) else 1e38
        b_min = b_bounds[0] if not np.isinf(b_bounds[0]) else 0.1
        b_max = b_bounds[1] if not np.isinf(b_bounds[1]) else 3.0
        
        bounds = ([A_min, b_min, n_bound_min], 
                  [A_max, b_max, n_bound_max])
        
        logger.debug(f"Fitting power law with initial params: A={A0}, b={b0}, n={n0}")
        
        # First step of curve fitting
        popt1, _ = curve_fit(power_law_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # Extract parameters from first fit
        _, _, n1 = popt1
        
        # Refine noise bounds for second step using the same noise_bound_percent
        n_bound_min_2 = n1 * (1 - noise_bound_percent)
        n_bound_max_2 = n1 * (1 + noise_bound_percent)
        
        # Second step with refined bounds
        bounds2 = ([A_min, b_min, n_bound_min_2], 
                   [A_max, b_max, n_bound_max_2])
        
        # Perform second fit using first fit result as initial guess
        popt2, _ = curve_fit(power_law_fn, x, y, p0=popt1, bounds=bounds2, maxfev=maxfev)
        
        # Log the fit results
        logger.info(f"Power law fit results: A={popt2[0]:.3e}, b={popt2[1]:.3f}, n={popt2[2]:.3e}")
        
        # Return the fitted parameters from second step
        return tuple(popt2)
    
    except Exception as e:
        logger.error(f"Error fitting power law: {str(e)}")
        # Return default values in case of fitting error
        return (1.0, 1.0, n0 if 'n0' in locals() else 0.01)


def fit_bending_power_law(
    x: np.ndarray,
    y: np.ndarray,
    noise_bound_percent: float = 0.7,
    maxfev: int = 1000000,
    initial_params: Optional[Dict[str, float]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[float, float, float, float]:
    """
    Fit a bending power law model to frequency data
    
    This function fits the model PSD(f) = A / [f(1+(f/fb)^(sh-1))] + n to the
    provided frequency and PSD data, using a two-step approach with curve_fit from scipy.optimize.
    This improves parameter estimation accuracy, especially for the noise level.
    
    Args:
        x: Frequency array
        y: PSD values array
        noise_bound_percent: Controls the allowed range for the noise parameter 
                           relative to the initial estimate (default: 0.7)
        maxfev: Maximum number of function evaluations (default: 1000000)
        initial_params: Optional dictionary with custom initial parameter guesses:
                - 'A': Amplitude
                - 'fb': Break frequency
                - 'sh': Shape parameter
        param_bounds: Optional dictionary with custom parameter bounds:
                - 'A': (min, max) for amplitude
                - 'fb': (min, max) for break frequency
                - 'sh': (min, max) for shape parameter
    
    Returns:
        Tuple of fitted parameters: (A, fb, sh, n):
            - A: Normalization constant
            - fb: Break frequency
            - sh: Shape parameter
            - n: Frequency-independent noise level
    
    Example:
        >>> x = np.logspace(-5, -1, 100)
        >>> y = 1.0 / (x * (1 + (x/0.01)**(1.5-1))) + 0.01
        >>> A, fb, sh, n = fit_bending_power_law(x, y)
        >>> print(f"A={A:.3e}, fb={fb:.3e}, sh={sh:.2f}, n={n:.3e}")
    """
    
    try:
        # Default values
        initial_params = initial_params or {}
        param_bounds = param_bounds or {}
        
        # Estimate the noise level from the high-frequency end of the spectrum
        n0 = float(np.mean(y[-10:]))
        
        # Estimate the break frequency (fb) as the frequency where PSD drops to 1/10 of max
        default_fb0 = x[np.argmin(np.abs(y - max(y) / 10))] if np.any(y < max(y) / 10) else x[len(x)//2]
        
        # Set initial guesses for parameters
        A0 = initial_params.get('A', 10.0)
        fb0 = initial_params.get('fb', default_fb0)
        sh0 = initial_params.get('sh', 1.0)
        
        p0 = [A0, fb0, sh0, n0]
        
        # Set bounds for the noise parameter based on noise_bound_percent
        n_bound_min = n0 * (1 - noise_bound_percent)
        n_bound_max = n0 * (1 + noise_bound_percent)
        
        # Set bounds for all parameters
        A_bounds = param_bounds.get('A', (0, 1e38))
        fb_bounds = param_bounds.get('fb', (x[0], x[-1]))
        sh_bounds = param_bounds.get('sh', (0.3, 3.0))
        
        # Handle infinity in bounds
        A_min = A_bounds[0] if not np.isinf(A_bounds[0]) else 0.0
        A_max = A_bounds[1] if not np.isinf(A_bounds[1]) else 1e38
        fb_min = fb_bounds[0] if not np.isinf(fb_bounds[0]) else x[0]
        fb_max = fb_bounds[1] if not np.isinf(fb_bounds[1]) else x[-1]
        sh_min = sh_bounds[0] if not np.isinf(sh_bounds[0]) else 0.3
        sh_max = sh_bounds[1] if not np.isinf(sh_bounds[1]) else 3.0
        
        bounds = ([A_min, fb_min, sh_min, n_bound_min], 
                  [A_max, fb_max, sh_max, n_bound_max])
        
        logger.debug(f"Fitting bending power law with initial params: A={A0}, fb={fb0}, sh={sh0}, n={n0}")
        
        # First step of curve fitting
        popt1, _ = curve_fit(bending_power_law_fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        
        # Extract parameters from first fit
        _, _, _, n1 = popt1
        
        # Refine noise bounds for second step
        n_bound_min_2 = n1 * (1 - noise_bound_percent)
        n_bound_max_2 = n1 * (1 + noise_bound_percent)
        
        # Second step with refined bounds
        bounds2 = ([A_min, fb_min, sh_min, n_bound_min_2], 
                   [A_max, fb_max, sh_max, n_bound_max_2])
        
        # Perform second fit using first fit result as initial guess
        popt2, _ = curve_fit(bending_power_law_fn, x, y, p0=popt1, bounds=bounds2, maxfev=maxfev)
        
        # Log the fit results
        logger.info(
            f"Bending power law fit results: "
            f"A={popt2[0]:.3e}, fb={popt2[1]:.3e}, sh={popt2[2]:.3f}, n={popt2[3]:.3e}"
        )
        
        # Return the fitted parameters from second step
        return tuple(popt2)
    
    except RuntimeError as e:
        logger.error(f"Curve fitting failed: {str(e)}")
        raise Exception(f"Curve fitting failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error fitting bending power law: {str(e)}")
        raise Exception(f"Error fitting bending power law: {str(e)}")


# ==========================================
# Quality Metrics
# ==========================================

# def calculate_fit_quality(
#     x: np.ndarray,
#     y: np.ndarray,
#     y_fit: np.ndarray
# ) -> Dict[str, float]:
#     """
#     Calculate goodness of fit metrics
    
#     Args:
#         x: Frequency array
#         y: Observed PSD values
#         y_fit: Fitted PSD values
    
#     Returns:
#         Dictionary with fit quality metrics
    
#     Example:
#         >>> x = np.linspace(0.01, 1, 100)
#         >>> y_obs = 1.0 / x**1.5 + 0.01
#         >>> y_fit = power_law_fn(x, 1.0, 1.5, 0.01)
#         >>> metrics = calculate_fit_quality(x, y_obs, y_fit)
#         >>> print(f"R²={metrics['r_squared']:.4f}")
#     """
    
#     # R-squared (coefficient of determination)
#     ss_res = np.sum((y - y_fit) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     r_squared = 1 - (ss_res / ss_tot)
    
#     # Reduced chi-squared
#     residuals = y - y_fit
#     chi_squared = np.sum((residuals / y) ** 2)
#     n_data = len(y)
#     n_params = 3  # Adjust based on model
#     reduced_chi_squared = chi_squared / (n_data - n_params)
    
#     # Root Mean Square Error
#     rmse = np.sqrt(np.mean(residuals ** 2))
    
#     # Mean Absolute Percentage Error
#     mape = np.mean(np.abs(residuals / y)) * 100
    
#     return {
#         "r_squared": float(r_squared),
#         "reduced_chi_squared": float(reduced_chi_squared),
#         "rmse": float(rmse),
#         "mape": float(mape),
#         "max_residual": float(np.max(np.abs(residuals))),
#         "mean_residual": float(np.mean(residuals))
#     }


# def calculate_residuals(
#     x: np.ndarray,
#     y: np.ndarray,
#     model_fn,
#     params: Tuple
# ) -> Dict[str, np.ndarray]:
#     """
#     Calculate residuals and their statistics
    
#     Args:
#         x: Frequency array
#         y: Observed PSD values
#         model_fn: Model function (power_law_fn or bending_power_law_fn)
#         params: Model parameters
    
#     Returns:
#         Dictionary with residual arrays
#     """
    
#     y_fit = model_fn(x, *params)
    
#     # Absolute residuals
#     abs_residuals = y - y_fit
    
#     # Relative residuals (multiplicative)
#     rel_residuals = y / y_fit
    
#     # Normalized residuals (in units of standard deviation)
#     std_residuals = abs_residuals / np.std(abs_residuals)
    
#     return {
#         "absolute": abs_residuals,
#         "relative": rel_residuals,
#         "normalized": std_residuals,
#         "fitted_values": y_fit
#     }