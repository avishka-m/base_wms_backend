"""
Seasonal Inventory Prediction System

AI-powered seasonal inventory forecasting using Facebook Prophet
and comprehensive data integration from multiple sources.
"""

__version__ = "1.0.0"
__author__ = "WMS AI Team"

# Main imports
from .data_collection import KaggleDataDownloader, WebDataScraper, WMSDataExtractor

__all__ = [
    "KaggleDataDownloader",
    "WebDataScraper",
    "WMSDataExtractor"
]
