# app/tools/__init__.py

"""
FITS Analysis Tools

Core utilities for FITS file analysis including:
- Statistics computation
- Power Spectral Density (PSD) analysis
- Model fitting (power law, bending power law)
- Visualization
"""

from app.tools.statistics import (
    calculate_statistics,
    calculate_percentiles,
    calculate_quantiles,
    detect_outliers,
    calculate_distribution_info
)

from app.tools.psd import (
    compute_psd,
    bin_psd,
    compute_psd_with_window,
    compute_psd_welch,
    validate_frequency_bounds,
    get_frequency_info
)

from app.tools.fitting import (
    power_law_fn,
    bending_power_law_fn,
    fit_power_law,
    fit_bending_power_law,
    # calculate_fit_quality,
    # calculate_residuals
)

from app.tools.plotting import (
    plot_psd_figure,
    plot_power_law_with_residual_figure,
    plot_bending_power_law_with_residual_figure
)

__all__ = [
    # Statistics
    'calculate_statistics',
    'calculate_percentiles',
    'calculate_quantiles',
    'detect_outliers',
    'calculate_distribution_info',
    
    # PSD
    'compute_psd',
    'bin_psd',
    'compute_psd_with_window',
    'compute_psd_welch',
    'validate_frequency_bounds',
    'get_frequency_info',
    
    # Fitting
    'power_law_fn',
    'bending_power_law_fn',
    'fit_power_law',
    'fit_bending_power_law',
    # 'calculate_fit_quality',
    # 'calculate_residuals',
    
    # Plotting
    'plot_psd_figure',
    'plot_power_law_with_residual_figure',
    'plot_bending_power_law_with_residual_figure'
]
