"""Shared pytest fixtures."""

import io
import math

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_jpeg() -> bytes:
    """64×64 checkerboard JPEG for optics/features tests."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for r in range(64):
        for c in range(64):
            if (r // 8 + c // 8) % 2 == 0:
                img[r, c] = 255
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


@pytest.fixture
def simple_K() -> np.ndarray:
    """Simple 640×480 intrinsic matrix (fx=fy=500, cx=320, cy=240)."""
    from autocal.optics.camera import intrinsic_matrix
    return intrinsic_matrix(500.0, 500.0, 320.0, 240.0)
