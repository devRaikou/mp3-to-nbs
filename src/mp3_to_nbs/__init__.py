"""
mp3-to-nbs — Convert MP3 audio files to Minecraft Note Block Studio (.nbs) format.

This package provides tools for analyzing audio files and converting them into
the NBS format used by Minecraft Note Block Studio (OpenNBS).
"""

__version__ = "0.1.0"
__author__ = "raikou"

from mp3_to_nbs.converter import convert, ConversionResult
from mp3_to_nbs.config import ConversionConfig

__all__ = ["convert", "ConversionResult", "ConversionConfig", "__version__"]
