"""
multi-agent-fits-dev-02/app/tools/plotting.py

Visualization tools for FITS data analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging

from app.tools.fitting import power_law_fn, bending_power_law_fn

logger = logging.getLogger(__name__)


def plot_psd_figure(
    x: np.ndarray, 
    y: np.ndarray, 
    title: str = "Power Spectral Density"
) -> Figure:
    """
    Create a figure displaying a Power Spectral Density (PSD) plot
    
    This function creates a standardized PSD plot with logarithmic axes,
    which is the conventional way to visualize PSDs in astrophysics.
    The logarithmic scale helps visualize power-law behavior as linear relationships
    and covers many orders of magnitude in both frequency and power.
    
    Args:
        x: Frequency array
        y: PSD values array
        title: Plot title
    
    Returns:
        Matplotlib Figure object containing the PSD plot
    
    Example:
        >>> x = np.logspace(-5, -1, 1000)
        >>> y = 1.0 / x**1.5
        >>> fig = plot_psd_figure(x, y, title="My PSD")
    """
    
    # Create figure and axes with specified size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the PSD data with logarithmic x and y axes
    ax.loglog(x, y, color='black', linewidth=1.5)
    
    # Add axis labels
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability on log-log plot
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Adjust layout to prevent labels from being cut off
    fig.tight_layout()
    
    logger.debug(f"PSD plot created: {title}")
    
    return fig


def plot_power_law_with_residual_figure(
    x: np.ndarray,
    y: np.ndarray,
    A: float,
    b: float,
    n: float,
    title: str = "Power Law Fit",
    yscale_range=None
) -> Figure:
    """
    Create a figure showing data with a power law model fit and residuals
    
    This function creates a plot displaying both the original PSD data
    and the fitted power law model on logarithmic axes, along with the residuals
    to help assess the quality of the fit. The fitted model
    parameters are included in the legend for easy reference.
    
    Args:
        x: Frequency array
        y: PSD values array
        A: Amplitude parameter from the fit
        b: Power law index from the fit
        n: Frequency-independent noise level from the fit
        title: Plot title
        yscale_range: Optional tuple of (ymin, ymax) to override automatic y-axis limits
    
    Returns:
        Matplotlib Figure object containing the data, fit, and residuals
    
    Example:
        >>> x = np.logspace(-5, -1, 100)
        >>> y = 1.0 / x**1.5 + 0.01
        >>> A, b, n = 1.0, 1.5, 0.01
        >>> fig = plot_power_law_with_residual_figure(x, y, A, b, n)
    """
    
    # Calculate model components
    y_fit = power_law_fn(x, A, b, n)       # Full model (signal + noise)
    y_model = power_law_fn(x, A, b, 0)     # Signal component only
    y_noise = np.ones_like(x) * n          # Noise component only
    
    # Calculate residuals as ratio of data to model (multiplicative residuals)
    residuals = y / y_fit
    
    # Create figure with two subplots (main plot and residuals)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    
    # ==========================================
    # Main Plot
    # ==========================================
    ax1.plot(x, y, color='black', label='Original PSD', linewidth=1.5)
    ax1.plot(x, y_fit, '--', color='red', label='PSD + Noise', linewidth=2)
    ax1.plot(x, y_model, '-', color='blue', label=rf'$P(f)=A f^{{-b}}+n$', linewidth=2)
    ax1.plot(x, y_noise, ':', color='green', label='Noise', linewidth=2)
    
    # Configure main plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Power Spectral Density', fontsize=12)
    ax1.set_title(f'{title}\nA={A:.2e}, b={b:.2f}, n={n:.2e}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Set y-axis limits
    if yscale_range is None:
        y_min = np.min(y[y > 0])
        y_max = np.max(y)
        y_lower = y_min * 0.5
        y_upper = y_max * 1.2
        ax1.set_ylim(y_lower, y_upper)
    else:
        ax1.set_ylim(yscale_range)
    
    # ==========================================
    # Residual Plot
    # ==========================================
    ax2.plot(x, residuals, '.', color='purple', markersize=4)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
    
    # Configure residual plot
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Residual\n(PSD/Model)', fontsize=10)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    logger.debug(f"Power law plot with residuals created: {title}")
    
    return fig


def plot_bending_power_law_with_residual_figure(
    x: np.ndarray,
    y: np.ndarray,
    A: float,
    fb: float,
    sh: float,
    n: float,
    title: str = "Bending Power Law Fit",
    yscale_range=None
) -> Figure:
    """
    Create a figure showing data with a bending power law model fit and residuals
    
    This function creates a plot displaying both the original PSD data and
    the fitted bending power law model on logarithmic axes, along with residuals
    to help assess the quality of the fit. The fitted model parameters
    are included in the legend for easy reference.
    
    A bending power law is characterized by different slopes at low and high
    frequencies, with a smooth transition at the break frequency (fb). This
    type of model is often used for accreting systems with characteristic
    timescales.
    
    Args:
        x: Frequency array
        y: PSD values array
        A: Amplitude parameter from the fit
        fb: Break frequency from the fit
        sh: Shape parameter from the fit
        n: Frequency-independent noise level from the fit
        title: Plot title
        yscale_range: Optional tuple of (ymin, ymax) to override automatic y-axis limits
    
    Returns:
        Matplotlib Figure object containing the data, fit, and residuals
    
    Example:
        >>> x = np.logspace(-5, -1, 100)
        >>> y = 1.0 / (x * (1 + (x/0.01)**(1.5-1))) + 0.01
        >>> A, fb, sh, n = 1.0, 0.01, 1.5, 0.01
        >>> fig = plot_bending_power_law_with_residual_figure(x, y, A, fb, sh, n)
    """
    
    # Calculate model components
    y_fit = bending_power_law_fn(x, A, fb, sh, n)  # Full model
    y_model = bending_power_law_fn(x, A, fb, sh, 0)  # Signal only
    y_noise = np.ones_like(x) * n  # Noise only
    
    # Calculate residuals
    residuals = y / y_fit
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 8),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )
    
    # ==========================================
    # Main Plot
    # ==========================================
    ax1.plot(x, y, color='black', label='Original PSD', linewidth=1.5)
    ax1.plot(x, y_fit, '--', color='red', label='PSD + Noise', linewidth=2)
    ax1.plot(
        x, y_model, '-', color='blue',
        label=rf'$P(f)=\frac{{A}}{{f[1+(f/f_b)^{{{sh:.1f}-1}}]}}$',
        linewidth=2
    )
    ax1.plot(x, y_noise, ':', color='green', label='Noise', linewidth=2)
    
    # Mark break frequency
    ax1.axvline(fb, color='orange', linestyle=':', linewidth=1.5, label=f'fb={fb:.2e}')
    
    # Configure main plot
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Power Spectral Density', fontsize=12)
    ax1.set_title(
        f'{title}\nA={A:.2e}, fb={fb:.2e}, sh={sh:.2f}, n={n:.2e}',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Set y-axis limits
    if yscale_range is None:
        y_min = np.min(y[y > 0])
        y_max = np.max(y)
        y_lower = y_min * 0.5
        y_upper = y_max * 1.2
        ax1.set_ylim(y_lower, y_upper)
    else:
        ax1.set_ylim(yscale_range)
    
    # ==========================================
    # Residual Plot
    # ==========================================
    ax2.plot(x, residuals, '.', color='purple', markersize=4)
    ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(fb, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Configure residual plot
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Residual\n(PSD/Model)', fontsize=10)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    logger.debug(f"Bending power law plot with residuals created: {title}")
    
    return fig