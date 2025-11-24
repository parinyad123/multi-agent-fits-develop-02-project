# multi-agent-fits-dev-02/app/services/astrosage/metadata_formatter.py

"""
Smart Metadata Formatter for AstroSage
Provides context-aware, minimal metadata summaries

Target: 150-250 tokens per analysis
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class MetadataFormatter:
    """
    Format metadata into compact, context-aware summaries
    
    Principles:
    - Show only RELEVANT info for the query/routing
    - Use compact notation (abbreviations, scientific notation)
    - Group related information
    - Highlight anomalies/important values
    """

    @staticmethod
    def format_compact_context(
        metadata: Dict[str, Any],
        routing_strategy: str,
        fitted_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate compact metadata context based on routing
        
        Args:
            metadata: Full metadata dict
            routing_strategy: "analysis", "astrosage", or "mixed"
            fitted_params: Fitted parameters (for smart context)
        
        Returns:
            Compact metadata string (~150-250 tokens)
        """
        
        if not metadata:
            return ""
        
        lines = []
        
        # ============================================
        # SECTION 1: Source & Observation (always show)
        # ============================================
        source_summary = MetadataFormatter._format_source_info(metadata)
        if source_summary:
            lines.append(source_summary)
        
        # ============================================
        # SECTION 2: Timing Context (for PSD/fitting queries)
        # ============================================
        if routing_strategy in ["analysis", "mixed"]:
            timing_summary = MetadataFormatter._format_timing_context(
                metadata, 
                fitted_params
            )
            if timing_summary:
                lines.append(timing_summary)
        
        # ============================================
        # SECTION 3: Source Context (for interpretation)
        # ============================================
        if routing_strategy in ["mixed", "astrosage"]:
            source_context = MetadataFormatter._format_source_context(
                metadata,
                fitted_params
            )
            if source_context:
                lines.append(source_context)
        
        return "\n".join(lines)

    @staticmethod
    def _format_source_info(metadata: Dict[str, Any]) -> str:
        """
        Format basic source information (1-2 lines)
        
        Example:
        Source: IRAS 13224-3809 | XMM/EPIC-pn | 0.3-10 keV | 14.2 hrs
        """
        
        crit = metadata.get('critical_fields', {})
        deriv = metadata.get('derived_quantities', {})
        
        parts = []
        
        # Source name
        source = crit.get('source_name', 'Unknown')
        parts.append(f"**Source:** {source}")
        
        # Instrument
        telescope = crit.get('telescope', '?')
        instrument = crit.get('instrument', '?')
        parts.append(f"{telescope}/{instrument}")
        
        # Energy band
        energy_band = deriv.get('energy_band_short', 'N/A')
        if energy_band != 'N/A':
            parts.append(energy_band)
        
        # Duration
        duration = deriv.get('observation_duration_hours', 0)
        if duration > 0:
            parts.append(f"{duration:.1f}h")
        
        return " | ".join(parts)
    
    @staticmethod
    def _format_timing_context(
        metadata: Dict[str, Any],
        fitted_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format timing constraints with smart interpretation
        
        Example:
        Timing: Δt=7.3ms → fₙ=68Hz | Obs=14h → fₘᵢₙ=2e-5Hz
        → Your fb=4e-5Hz is 2× fₘᵢₙ (well-resolved) ✓
        """
        
        deriv = metadata.get('derived_quantities', {})
        crit = metadata.get('critical_fields', {})
        
        lines = []
        parts = []
        
        # Time resolution
        dt = crit.get('time_bin_size', 0)
        f_nyq = deriv.get('nyquist_frequency_hz', 0)
        
        if dt > 0 and f_nyq > 0:
            # Format dt nicely
            if dt < 1:
                dt_str = f"{dt*1000:.1f}ms"
            else:
                dt_str = f"{dt:.2f}s"
            
            parts.append(f"Δt={dt_str} → fₙ={f_nyq:.0f}Hz")
        
        # Duration
        duration = deriv.get('observation_duration_seconds', 0)
        f_min = deriv.get('min_frequency_hz', 0)
        
        if duration > 0 and f_min > 0:
            # Convert duration to readable format
            if duration < 3600:
                dur_str = f"{duration/60:.0f}min"
            else:
                dur_str = f"{duration/3600:.1f}h"
            
            parts.append(f"Obs={dur_str} → fₘᵢₙ={f_min:.0e}Hz")
        
        if parts:
            lines.append("**Timing:** " + " | ".join(parts))
        
        # ============================================
        # Smart interpretation (if fitted params available)
        # ============================================
        if fitted_params:
            interpretation = MetadataFormatter._interpret_timing(
                fitted_params,
                f_min,
                f_nyq,
                duration
            )
            if interpretation:
                lines.append(f"→ {interpretation}")
        
        return "\n".join(lines) if lines else ""
    
    @staticmethod
    def _interpret_timing(
        fitted_params: Dict[str, Any],
        f_min: float,
        f_nyq: float,
        duration: float
    ) -> str:
        """
        Smart interpretation of fitted parameters vs timing constraints
        
        Checks:
        - Is break frequency well-resolved?
        - Is it too close to Nyquist?
        - How many characteristic cycles observed?
        """
        
        interpretations = []
        
        # Check bending power law break frequency
        if 'bending_power_law' in fitted_params:
            bpl = fitted_params['bending_power_law'].get('fitted_parameters', {})
            fb = bpl.get('fb', 0)
            
            if fb > 0 and f_min > 0 and f_nyq > 0:
                # Check if well-resolved
                if fb > 3 * f_min:
                    ratio = fb / f_min
                    interpretations.append(
                        f"fb={fb:.1e}Hz is {ratio:.0f}× fₘᵢₙ (well-resolved ✓)"
                    )
                elif fb < 2 * f_min:
                    interpretations.append(
                        f"fb={fb:.1e}Hz ≈ fₘᵢₙ (edge of resolution ⚠️)"
                    )
                
                # Check Nyquist distance
                if fb > 0.5 * f_nyq:
                    interpretations.append(
                        f"fb near Nyquist (may have aliasing ⚠️)"
                    )
                
                # Number of cycles
                if duration > 0:
                    t_break = 1.0 / fb
                    n_cycles = duration / t_break
                    
                    if n_cycles > 10:
                        interpretations.append(
                            f"{n_cycles:.0f} cycles observed"
                        )
                    elif n_cycles < 3:
                        interpretations.append(
                            f"Only {n_cycles:.1f} cycles (limited statistics ⚠️)"
                        )
        
        return " | ".join(interpretations)
    
    @staticmethod
    def _format_source_context(
        metadata: Dict[str, Any],
        fitted_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format source context with literature comparison
        
        Example:
        Context: NLS1 AGN (M_BH≈2e6 M☉)
        Literature: b∈[1.2,1.8], fb∈[1e-4,1e-3]Hz typical
        → Your b=0.80 is LOWER (flatter spectrum)
        """
        
        ctx = metadata.get('source_context', {})
        
        if not ctx.get('is_known_source'):
            return ""
        
        lines = []
        parts = []
        
        # Source type
        source_type = ctx.get('type', 'Unknown')
        parts.append(source_type)
        
        # Black hole mass
        bh_mass = ctx.get('black_hole_mass_solar', 0)
        if bh_mass > 0:
            if bh_mass >= 1e6:
                mass_str = f"{bh_mass/1e6:.0f}e6"
            else:
                mass_str = f"{bh_mass:.0e}"
            parts.append(f"M_BH≈{mass_str}M☉")
        
        if parts:
            lines.append("**Context:** " + " | ".join(parts))
        
        # ============================================
        # Literature comparison (if fitted params available)
        # ============================================
        if fitted_params and 'typical_psd' in ctx:
            typical = ctx['typical_psd']
            comparison = MetadataFormatter._compare_with_literature(
                fitted_params,
                typical
            )
            if comparison:
                lines.append(f"**Literature:** {comparison}")
        
        return "\n".join(lines) if lines else ""
    
    @staticmethod
    def _compare_with_literature(
        fitted_params: Dict[str, Any],
        typical_psd: Dict[str, Any]
    ) -> str:
        """
        Compare fitted values with literature
        
        Returns compact comparison string
        """
        
        comparisons = []
        
        # Power law index
        if 'power_law' in fitted_params:
            pl = fitted_params['power_law'].get('fitted_parameters', {})
            b = pl.get('b', 0)
            
            b_range = typical_psd.get('power_law_index_range', [])
            if b > 0 and b_range:
                b_min, b_max = b_range
                
                if b < b_min:
                    comparisons.append(
                        f"b={b:.2f} < typical [{b_min}-{b_max}] (flatter)"
                    )
                elif b > b_max:
                    comparisons.append(
                        f"b={b:.2f} > typical [{b_min}-{b_max}] (steeper)"
                    )
                else:
                    comparisons.append(
                        f"b={b:.2f} ∈ [{b_min}-{b_max}] (typical)"
                    )
        
        # Break frequency
        if 'bending_power_law' in fitted_params:
            bpl = fitted_params['bending_power_law'].get('fitted_parameters', {})
            fb = bpl.get('fb', 0)
            
            fb_range = typical_psd.get('break_frequency_range', [])
            if fb > 0 and fb_range:
                fb_min, fb_max = fb_range
                
                if fb < fb_min:
                    comparisons.append(
                        f"fb={fb:.1e}Hz < typical (longer timescales)"
                    )
                elif fb > fb_max:
                    comparisons.append(
                        f"fb={fb:.1e}Hz > typical (shorter timescales)"
                    )
                else:
                    comparisons.append(
                        f"fb={fb:.1e}Hz ∈ typical range"
                    )
        
        return " | ".join(comparisons)
    
    @staticmethod
    def format_for_analysis_only(metadata: Dict[str, Any]) -> str:
        """
        Ultra-minimal format for ANALYSIS routing
        Just the essentials
        
        Example:
        IRAS 13224-3809 | XMM/EPIC-pn | soft X-ray | 14h
        """
        
        crit = metadata.get('critical_fields', {})
        deriv = metadata.get('derived_quantities', {})
        
        source = crit.get('source_name', 'Unknown')
        telescope = crit.get('telescope', '?')
        instrument = crit.get('instrument', '?')
        energy = deriv.get('energy_band_short', '?')
        duration = deriv.get('observation_duration_hours', 0)
        
        parts = [
            source,
            f"{telescope}/{instrument}",
            energy,
            f"{duration:.1f}h" if duration > 0 else "?"
        ]
        
        return " | ".join(parts)


# ============================================
# Helper: Extract fitted parameters structure
# ============================================
def extract_fitted_params(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract fitted parameters in consistent format
    
    Returns:
    {
        "power_law": {"fitted_parameters": {...}},
        "bending_power_law": {"fitted_parameters": {...}}
    }
    """
    
    params = {}
    
    if 'power_law' in analysis_results:
        params['power_law'] = analysis_results['power_law']
    
    if 'bending_power_law' in analysis_results:
        params['bending_power_law'] = analysis_results['bending_power_law']
    
    return params