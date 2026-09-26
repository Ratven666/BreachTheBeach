from __future__ import annotations


class WaveModelError(Exception):
    """Base exception for the waves module."""


class WaveInputError(WaveModelError):
    """Raised when input data is invalid or incomplete."""


class WaveConfigurationError(WaveModelError):
    """Raised when model/service configuration is invalid."""


class WaveBathymetryError(WaveModelError):
    """Raised when bathymetry-dependent wave calculations fail."""


__all__ = [
    "WaveModelError",
    "WaveInputError",
    "WaveConfigurationError",
    "WaveBathymetryError",
]