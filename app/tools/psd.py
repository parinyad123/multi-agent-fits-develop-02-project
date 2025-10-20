"""
multi-agent-fits-dev-02/app/tools/psd.py

Power Spectral Density (PSD) computation and binning utilities

https://chatgpt.com/c/68ef2407-3550-8322-9b4a-bc708a8fb2bf
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
https://cran.r-project.org/web/packages/psd/vignettes/normalization.pdf
"""

import numpy as np
from scipy.stats import binned_statistic
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def compute_psd(rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Power Spectral Density (PSD) from time series

    This function takes a time series (with assumed sampling interval dt=1)
    and computes its Power Spectral Density usong Fast Fourier transfrom (FTT).

    Arge:
        rate: Time series data array

    Returns:
        Tuple containing:
            - freqs: Array of frequencies (Hz)
            - psd: Array of corresponding power values

    Example:
        >>> rate = np.random.randn(10000)
        >>> freqs, psd = compute_psd(rate)
        >>> print(f"Frequency range: {freqs[0]:.6f} to {freqs[-1]:.6f} Hz")
        >>> print(f"PSD range: {psd.min():.6f} to {psd.max():.6f}")
    """

    N = len(rate)
    dt = 1.0        # Assumed time step between samples

    logger.debug(f"Computing PSD for {N} data points")

    # Calculate FFT of the entire spectrum
    fft_vals = np.fft.fft(rate)[:N//2+1]

    # Compute PSD by squaring the magnitude of FFT values and multiplying by 2
    psd = 2.0 * (np.abs(fft_vals) ** 2)
    
    # Calculate the corresponding frequencies
    freqs = np.fft.fftfreq(N, dt)[:N//2+1]
    
    # Remove DC component (first value) and Nyquist frequency (last value if it exists)
    # DC is removed because it just represents the average of the signal
    # Nyquist is at the edge of what can be represented given the sampling rate
    
    logger.debug(f"PSD computed: {len(freqs)-2} frequency bins (excluding DC and Nyquist)")
    
    return freqs[1:-1], psd[1:-1]

def bin_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    low_freq: float = 1e-5,
    high_freq: float = 0.05,
    bins: int = 3500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin PSD values into specified frequency ranges
    
    This function takes frequency and PSD arrays and groups them into a specified
    number of bins between the low and high frequency limits. This is useful for
    reducing noise in the PSD and making the data more manageable for analysis
    and visualization, especially for log-scale plots.
    
    This function ensures bin count does not exceed the number of data points and
    automatically adjusts if necessary.
    
    Args:
        freqs: Array of frequencies
        psd: Array of PSD values
        low_freq: Minimum frequency to include (default: 1e-5 Hz)
        high_freq: Maximum frequency to include (default: 0.05 Hz)
        bins: Number of frequency bins to use (default: 3500)
    
    Returns:
        Tuple containing:
            - centers: Array of bin center frequencies
            - psd_binned: Array of mean PSD values in each bin
    
    Example:
        >>> freqs, psd = compute_psd(rate)
        >>> x, y = bin_psd(freqs, psd, low_freq=1e-4, high_freq=0.1, bins=1000)
        >>> print(f"Binned to {len(x)} frequency bins")
    """
    
    # Filter frequencies within range
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    freqs_filtered = freqs[mask]
    psd_filtered = psd[mask]
    
    total_points = len(freqs_filtered)
    
    logger.debug(
        f"Binning PSD: {total_points} points in range "
        f"[{low_freq:.6e}, {high_freq:.6e}] Hz into {bins} bins"
    )
    
    # Check if bins exceed the number of data points and log a warning
    if bins >= total_points:
        logger.warning(
            f"Number of bins ({bins}) >= number of data points ({total_points}). "
            "Reducing bins to avoid empty bins or NaNs in PSD."
        )
    
    # Automatically reduce bins to prevent exceeding the number of points
    adjusted_bins = min(bins, total_points)
    
    # Create bin edges linearly spaced between low_freq and high_freq
    bin_edges = np.linspace(low_freq, high_freq, adjusted_bins + 1)
    
    # Perform binning: compute mean PSD value in each bin
    psd_binned, _, _ = binned_statistic(
        freqs_filtered, 
        psd_filtered, 
        statistic='mean', 
        bins=bin_edges
    )
    
    # Calculate bin centers (midpoint between edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    logger.debug(f"PSD binned: {len(centers)} bins created")
    
    return centers, psd_binned

def compute_psd_with_window(
    rate: np.ndarray,
    window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD with windowing function applied
    
    Windowing helps reduce spectral leakage in FFT analysis
    
    Args:
        rate: Time series data
        window: Window type ("hann", "hamming", "blackman", "bartlett", "none")
    
    Returns:
        Tuple of (frequencies, psd)
    
    Example:
        >>> rate = np.random.randn(10000)
        >>> freqs, psd = compute_psd_with_window(rate, window="hann")
    """
    
    N = len(rate)
    
    # Apply window
    if window == "hann":
        window_func = np.hanning(N)
    elif window == "hamming":
        window_func = np.hamming(N)
    elif window == "blackman":
        window_func = np.blackman(N)
    elif window == "bartlett":
        window_func = np.bartlett(N)
    elif window == "none":
        window_func = np.ones(N)
    else:
        logger.warning(f"Unknown window type: {window}, using Hann window")
        window_func = np.hanning(N)
    
    # Apply window to data
    windowed_rate = rate * window_func
    
    # Compute PSD
    dt = 1.0
    fft_vals = np.fft.fft(windowed_rate)[:N//2+1]
    psd = 2.0 * (np.abs(fft_vals) ** 2)
    freqs = np.fft.fftfreq(N, dt)[:N//2+1]
    
    # Normalize for window power
    window_power = np.sum(window_func ** 2) / N
    psd = psd / window_power
    
    logger.debug(f"PSD computed with {window} window")
    
    return freqs[1:-1], psd[1:-1]


def compute_psd_welch(
    rate: np.ndarray,
    segment_length: int = 1024,
    overlap: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD using Welch's method (averaged periodogram)
    
    Welch's method reduces noise by averaging multiple overlapping segments
    
    Args:
        rate: Time series data
        segment_length: Length of each segment
        overlap: Overlap fraction between segments (0-1)
    
    Returns:
        Tuple of (frequencies, psd)
    
    Example:
        >>> rate = np.random.randn(10000)
        >>> freqs, psd = compute_psd_welch(rate, segment_length=512, overlap=0.5)
    """
    
    from scipy import signal
    
    nperseg = segment_length
    noverlap = int(segment_length * overlap)
    
    freqs, psd = signal.welch(
        rate,
        fs=1.0,  # Sampling frequency (1 Hz)
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    
    logger.debug(
        f"PSD computed using Welch's method: "
        f"segment_length={segment_length}, overlap={overlap}"
    )
    
    # Remove DC and Nyquist
    return freqs[1:-1], psd[1:-1]


def get_frequency_info(freqs: np.ndarray) -> dict:
    """
    Get information about the frequency array for API response
    
    Args:
        freqs: Array of frequencies (Hz)
    
    Returns:
        Dictionary with frequency information
    
    Example:
        >>> freqs = np.linspace(1e-5, 0.5, 10000)
        >>> info = get_frequency_info(freqs)
        >>> print(info)
        {'min_freq': 1e-05, 'max_freq': 0.5, 'n_freq_points': 10000, ...}
    """
    
    return {
        "min_freq": float(np.min(freqs)),
        "max_freq": float(np.max(freqs)),
        "n_freq_points": len(freqs),
        "freq_resolution": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
        "nyquist_freq": float(np.max(freqs))
    }


def validate_frequency_bounds(
    low_freq: float,
    high_freq: float,
    freqs: np.ndarray
) -> Tuple[float, float]:
    """
    Validate and constrain frequency bounds to actual data limits
    
    Args:
        low_freq: User-requested minimum frequency (Hz)
        high_freq: User-requested maximum frequency (Hz)
        freqs: Array of actual frequencies (Hz) from the data
    
    Returns:
        Tuple of (constrained_low_freq, constrained_high_freq)
    
    Example:
        >>> freqs = np.linspace(1e-4, 0.1, 1000)
        >>> low, high = validate_frequency_bounds(1e-5, 0.5, freqs)
        >>> print(f"Validated: {low:.6e} to {high:.6e}")
    """
    
    # Get actual frequency limits from data
    min_freq = float(np.min(freqs))
    max_freq = float(np.max(freqs))
    
    # Check if low_freq > high_freq and swap if necessary
    if low_freq > high_freq:
        logger.warning(f"low_freq ({low_freq}) > high_freq ({high_freq}), swapping")
        low_freq, high_freq = high_freq, low_freq
    
    # Constrain to actual data range
    validated_low_freq = max(low_freq, min_freq)
    validated_high_freq = min(high_freq, max_freq)
    
    if validated_low_freq != low_freq or validated_high_freq != high_freq:
        logger.info(
            f"Frequency bounds adjusted: "
            f"[{low_freq:.6e}, {high_freq:.6e}] → "
            f"[{validated_low_freq:.6e}, {validated_high_freq:.6e}]"
        )
    
    return validated_low_freq, validated_high_freq