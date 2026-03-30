"""
Survival Model Module - Convenience re-export.

The canonical implementation lives in src.models.survival_analysis.
This module re-exports SurvivalModel for import compatibility.
"""

from src.models.survival_analysis import SurvivalModel

__all__ = ["SurvivalModel"]
