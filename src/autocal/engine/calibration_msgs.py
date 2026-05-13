"""MCAP JSON message dataclasses for the calibration engine output topics."""
from __future__ import annotations

import dataclasses

from autocal.io.mcap_reader import register_message


@register_message
@dataclasses.dataclass
class PoseErrorMsg:
    """Per-frame pose displacement written to /stats/pose_error."""
    position_error_m: float


@register_message
@dataclasses.dataclass
class CalibrationDeltaMsg:
    """Before/after calibration comparison written to /stats/calibration_delta."""
    fx_initial: float
    fx_optimized: float
    fx_delta: float
    fy_initial: float
    fy_optimized: float
    fy_delta: float
    cx_initial: float
    cx_optimized: float
    cx_delta: float
    cy_initial: float
    cy_optimized: float
    cy_delta: float
    k1_initial: float
    k1_optimized: float
    k1_delta: float
    k2_initial: float
    k2_optimized: float
    k2_delta: float
    p1_initial: float
    p1_optimized: float
    p1_delta: float
    p2_initial: float
    p2_optimized: float
    p2_delta: float
