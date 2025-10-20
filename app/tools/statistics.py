"""
multi-agent-fits-dev-02/app/tools/statistics.py

Statistical analysis tools for astronomical time series data
"""

import numpy as np
from typing import List, Dict, Any
import logging
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

def calculate_statistics(
        data: np.ndarray,
        metrics: List[str]
) -> Dict[str, Any]:
    """
    Calculate statistical metrics from time series data

    Args:
        data: Numpy array of time series data
        metrics: List of statistical metrics to calculate

    Available metrics:
        - Basic: "mean", "median", "std", "min", "max", "count"
        - Percentiles: "percentile_X" (e.g., "percentile_25", "percentile_90")
        - Quantiles: Will be handled by StatisticsCapability

    Returns:
        Dictionary mapping metrics to their calculated values
    
    Raises:
        TypeError: If data is not a numpy array
        ValueError: If data is None or empty
    """

    # ========================================
    # CRITICAL VALIDATION
    # ========================================
    logger.debug(f"calculate_statistics called with data type: {type(data)}")
    
    # Check for None
    if data is None:
        error_msg = "Data cannot be None"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check type
    if not isinstance(data, np.ndarray):
        error_msg = f"Data must be numpy.ndarray, got {type(data)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Check if empty
    if data.size == 0:  # ← ใช้ .size แทน len()
        error_msg = "Data array is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"✓ Data validation passed: shape={data.shape}, size={data.size}, dtype={data.dtype}")

    # Initialize result dictionary
    stats: Dict[str, Any] = {}

    # ==========================================
    # Basic Statistics
    # ==========================================

    try:
        if "mean" in metrics:
            stats["mean"] = float(np.mean(data))

        if "median" in metrics:
            stats["median"] = float(np.median(data))

        if "std" in metrics:
            stats["std"] = float(np.std(data))
        
        if "var" in metrics:
            stats["var"] = float(np.var(data))
        
        if "min" in metrics:
            stats["min"] = float(np.min(data))
        
        if "max" in metrics:
            stats["max"] = float(np.max(data))
        
        if "count" in metrics:
            stats["count"] = int(data.size)  # ← ใช้ .size
        
        if "sum" in metrics:
            stats["sum"] = float(np.sum(data))

        if "range" in metrics:
            stats["range"] = float(np.max(data) - np.min(data))

        if "iqr" in metrics:
            stats["iqr"] = float(np.percentile(data, 75) - np.percentile(data, 25))

        if "skewness" in metrics:
            stats["skewness"] = float(scipy_stats.skew(data))
        
        if "kurtosis" in metrics:
            stats["kurtosis"] = float(scipy_stats.kurtosis(data))

        if "cv" in metrics:
            mean_val = np.mean(data)
            stats["cv"] = float(np.std(data) / mean_val) if mean_val != 0 else None
            
    except Exception as e:
        logger.error(f"Error calculating basic statistics: {e}", exc_info=True)
        raise RuntimeError(f"Failed to calculate basic statistics: {str(e)}") from e

    # ==========================================
    # Percentiles
    # ==========================================

    for metric in metrics:
        if metric.startswith("percentile_"):
            try:
                # Extract percentile value from metric name
                percentile_value = float(metric.split("_")[1])

                if 0 <= percentile_value <= 100:
                    value = float(np.percentile(data, percentile_value))
                    stats[metric] = value
                else:
                    logger.warning(f"Invalid percentile value: {percentile_value} (must be 0-100)")

            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid percentile format: {metric}")

        if metric.startswith("quantile_"):
            try:
                quantile_value = float(metric.split("_")[1])

                if 0 <= quantile_value <= 1:
                    value = float(np.quantile(data, quantile_value))
                    stats[metric] = value
                else:
                    logger.warning(f"Invalid quantile: {quantile_value} (must be 0-1)")
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid quantile format: {metric}")

    logger.info(f"Statistics calculation completed: {len(stats)} metrics")
    return stats

def calculate_percentiles(
    data: np.ndarray,
    percentiles: List[float]
) -> Dict[str, float]:
    """
    Calculate multiple percentiles at once
    
    Args:
        data: NumPy array of data
        percentiles: List of percentile values (0-100)
    
    Returns:
        Dictionary mapping percentile names to values
    
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = calculate_percentiles(data, [25, 50, 75, 90])
        >>> print(result)
        {'percentile_25': 3.25, 'percentile_50': 5.5, 'percentile_75': 7.75, 'percentile_90': 9.1}
    """
    
    result = {}
    
    for p in percentiles:
        if 0 <= p <= 100:
            value = float(np.percentile(data, p))
            result[f"percentile_{p}"] = value
        else:
            logger.warning(f"Invalid percentile: {p} (must be 0-100)")
    
    return result


def calculate_quantiles(
    data: np.ndarray,
    quantiles: List[float]
) -> Dict[str, float]:
    """
    Calculate multiple quantiles at once
    
    Args:
        data: NumPy array of data
        quantiles: List of quantile values (0-1)
    
    Returns:
        Dictionary mapping quantile names to values
    
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> result = calculate_quantiles(data, [0.25, 0.5, 0.75, 0.9])
        >>> print(result)
        {'quantile_0_25': 3.25, 'quantile_0_5': 5.5, 'quantile_0_75': 7.75, 'quantile_0_9': 9.1}
    """
    
    result = {}
    
    for q in quantiles:
        if 0 <= q <= 1:
            value = float(np.quantile(data, q))
            # Format key: 0.25 → quantile_0_25
            q_key = f"quantile_{str(q).replace('.', '_')}"
            result[q_key] = value
        else:
            logger.warning(f"Invalid quantile: {q} (must be 0-1)")
    
    return result


def detect_outliers(
    data: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, Any]:
    """
    Detect outliers in data using various methods
    
    Args:
        data: NumPy array of data
        method: Detection method ("iqr", "zscore", "mad")
        threshold: Threshold for outlier detection
            - IQR method: typically 1.5 or 3.0
            - Z-score method: typically 3.0
            - MAD method: typically 3.0
    
    Returns:
        Dictionary with outlier information
    
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
        >>> outliers = detect_outliers(data, method="iqr")
        >>> print(outliers["n_outliers"])
        1
    """
    
    result = {
        "method": method,
        "threshold": threshold,
        "n_total": len(data),
        "n_outliers": 0,
        "outlier_indices": [],
        "outlier_values": [],
        "lower_bound": None,
        "upper_bound": None
    }
    
    if method == "iqr":
        # IQR method (Tukey's fences)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        result["lower_bound"] = float(lower_bound)
        result["upper_bound"] = float(upper_bound)
        result["iqr"] = float(iqr)
    
    elif method == "zscore":
        # Z-score method
        mean = np.mean(data)
        std = np.std(data)
        
        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold
        
        result["mean"] = float(mean)
        result["std"] = float(std)
    
    elif method == "mad":
        # Median Absolute Deviation method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Modified z-score using MAD
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        result["median"] = float(median)
        result["mad"] = float(mad)
    
    else:
        logger.error(f"Unknown outlier detection method: {method}")
        return result
    
    # Extract outlier information
    outlier_indices = np.where(outlier_mask)[0]
    result["n_outliers"] = int(len(outlier_indices))
    result["outlier_indices"] = outlier_indices.tolist()
    result["outlier_values"] = data[outlier_mask].tolist()
    result["outlier_percentage"] = float(len(outlier_indices) / len(data) * 100)
    
    return result


def calculate_distribution_info(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive distribution information
    
    Args:
        data: NumPy array of data
    
    Returns:
        Dictionary with distribution metrics
    """
    
    from scipy import stats as scipy_stats
    
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "mode": float(scipy_stats.mode(data, keepdims=True)[0][0]),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "q1": float(np.percentile(data, 25)),
        "q2": float(np.percentile(data, 50)),
        "q3": float(np.percentile(data, 75)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "skewness": float(scipy_stats.skew(data)),
        "kurtosis": float(scipy_stats.kurtosis(data)),
        "cv": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else None
    }