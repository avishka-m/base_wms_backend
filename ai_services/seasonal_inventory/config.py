
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from pathlib import Path

# Ensure required environment variables are set before importing base config
if "MONGODB_URL" not in os.environ:
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
if "DATABASE_NAME" not in os.environ:
    os.environ["DATABASE_NAME"] = "warehouse_management"

# Use absolute import for base config
from config.base import (
    MONGODB_URL, DATABASE_NAME, PROJECT_VERSION, ENVIRONMENT as BASE_ENVIRONMENT,
    get_database_config
)


# Load environment variables
load_dotenv()




PROJECT_NAME = "Seasonal Inventory Prediction"
PROJECT_DESCRIPTION = "AI-powered seasonal inventory forecasting using Facebook Prophet"


API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8003"))
API_PREFIX = "/api/v1"
API_TITLE = "Seasonal Inventory API"


# Database URLs inherited from base config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# WMS Integration
WMS_API_BASE_URL = os.getenv("WMS_API_BASE_URL", "http://localhost:8002/api/v1")
WMS_API_TIMEOUT = 30




# Data Paths
# DATA_DIR = "base_wms_backend/ai_services/seasonal_inventory/data"
# DATASETS_DIR = f"{DATA_DIR}/datasets"
# PROCESSED_DIR = "base_wms_backend/ai_services/seasonal_inventory/data/processed"
# MODELS_DIR = "base_wms_backend/ai_services/seasonal_inventory/data/models"
# CACHE_DIR = f"{DATA_DIR}/cache"

# Always resolve to absolute path relative to this file
_CONFIG_DIR = Path(__file__).parent.resolve()
DATA_DIR = str((_CONFIG_DIR / "data").resolve())
DATASETS_DIR = str((_CONFIG_DIR / "data" / "datasets").resolve())
PROCESSED_DIR = str((_CONFIG_DIR / "data" / "processed").resolve())
MODELS_DIR = str((_CONFIG_DIR / "data" / "models").resolve())
CACHE_DIR = str((_CONFIG_DIR / "data" / "cache").resolve())


# Prophet Parameters for Each Category
PROPHET_CONFIG = {
    "books_media": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    "clothing": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    "electronics": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    "health_beauty": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    "home_garden": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    "sports_outdoors": {
        "growth": "linear",
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    # Shared custom seasonalities and holidays
    "custom_seasonalities": [
        {
            "name": "monthly",
            "period": 30.5,
            "fourier_order": 5,
            "prior_scale": 10.0
        },
        {
            "name": "quarterly",
            "period": 91.25,
            "fourier_order": 3,
            "prior_scale": 10.0
        }
    ],
    "holidays": {
        "countries": ["US", "BR", "GB", "IN"],
        "custom_events": [
            {"holiday": "Black Friday", "date": "2024-11-29"},
            {"holiday": "Valentine's Day", "date": "2024-02-14"}
        ]
    }
}


# # Model Training Configuration (Standardized)
# TRAINING_CONFIG = {
#     "cross_validation": {
#         "initial_window": "730 days",   # Initial training window (2 years)
#         "period": "90 days",            # Retrain every 3 months
#         "horizon": "90 days"            # Forecast horizon for validation (3 months)
#     },
#     "hyperparameter_tuning": {
#         "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
#         "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
#         "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0]
#     },
#     "validation": {
#         "train_ratio": 0.8,              # 80% for training
#         "test_ratio": 0.1,               # 10% for testing
#         "min_train_days": 365            # At least 1 year of data for training
#     },
#     "early_stopping": {
#         "enabled": True,
#         "patience": 5,                   # Stop if no improvement after 5 rounds
#         "min_delta": 0.001               # Minimum improvement to continue
#     },
#     "random_seed": 42                    # For reproducibility
# }


# Feature Engineering Settings
# FEATURE_CONFIG = {
#     "temporal_features": [
#         "year", "month", "day", "dayofweek", "quarter",
#         "is_weekend", "is_month_start", "is_month_end",
#         "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
#     ],
    
#     "lag_features": {
#         "periods": [7, 14, 30, 90, 365],
#         "stats": ["mean", "std", "min", "max", "median"]
#     },
    
#     "rolling_features": {
#         "windows": [7, 14, 30, 90],
#         "stats": ["mean", "std", "trend"]
#     },
    
#     "external_features": [
#         "temperature", "precipitation", "humidity",
#         "consumer_price_index", "unemployment_rate",
#         "stock_market_index"
#     ]
# }

# Data Quality Checks
DATA_QUALITY = {
    "min_data_points": 100,
    "max_missing_ratio": 0.3,
    "outlier_threshold": 3.0,  # Standard deviations
    "min_forecast_horizon": 7,
    "max_forecast_horizon": 365
}



# Dashboard Configuration
DASHBOARD_CONFIG = {
    "refresh_interval": 300,  # 5 minutes
    "default_forecast_days": 90,
    "max_forecast_days": 365,
    "chart_height": 400,
    "chart_width": 800,
    
    "colors": {
        "forecast": "#1f77b4",
        "actual": "#ff7f0e", 
        "upper_bound": "#2ca02c",
        "lower_bound": "#d62728",
        "trend": "#9467bd",
        "seasonal": "#8c564b"
    }
}

# Plot Settings
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "font_size": 12,
    "title_size": 16,
    "save_format": "png"
}



# Caching Configuration
CACHE_CONFIG = {
    "forecast_cache_ttl": 3600,  # 1 hour
    "data_cache_ttl": 1800,      # 30 minutes
    "model_cache_ttl": 86400,    # 24 hours
    "redis_key_prefix": "seasonal_inventory:",
    "max_cache_size": "500MB"
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "max_workers": 4,
    "batch_size": 1000,
    "chunk_size": 10000,
    "parallel_forecasts": True,
    "use_gpu": False  # Set to True if CUDA available
}



# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "logs/seasonal_inventory.log",
    "max_file_size": "10MB",
    "backup_count": 5,
    "rotation": "1 week"
}

# Monitoring Metrics
MONITORING_CONFIG = {
    "prometheus_port": 8004,
    "metrics_endpoint": "/metrics",
    "health_check_endpoint": "/health",
    "alert_thresholds": {
        "forecast_accuracy_min": 0.7,
        "api_response_time_max": 5.0,
        "error_rate_max": 0.05
    }
}



# Business Configuration
BUSINESS_CONFIG = {
    "inventory_thresholds": {
        #if current stock is 100 units
        "safety_stock_multiplier": 1.5,#to handle sudden demand spike or delay,if lead time is 1 week , keep 100*1.5 =150 as the safety stock.
        "reorder_point_multiplier": 2.0,#re-order when stock drop below ....(100*2=200)
        "max_stock_multiplier": 4.0,#do not lead stock exceed(100*4=400), it would be oer stocking.
         "stockout_probability_threshold": 0.05, 
        #  If forecast shows more than 5% chance of running out of stock, trigger alerts or restocking
    },
    
    # "seasonality_detection": {
    #     "min_seasonal_strength": 0.3,
    #     "trend_change_threshold": 0.1,
    #     "seasonal_periods": [7, 30, 90, 365],
    # },
    
    "forecast_validation": {
        "max_growth_rate": 2.0,  # 200% growth
        # "min_accuracy_score": 0.6,
        # "confidence_interval_coverage": 0.8
    }
}

# Alert Configuration
ALERT_CONFIG = {
    "email_notifications": True,
    "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),#Send alert messages to a Slack channel using this URL.
    "alert_cooldown": 3600,  # 1 hour,Wait 1 hour between alerts of the same type to avoid spamming.
    
    "alert_types": {
        "low_accuracy": {"threshold": 0.7, "severity": "medium"},
        "high_forecast_error": {"threshold": 0.3, "severity": "high"},
        "data_quality_issues": {"threshold": 0.8, "severity": "medium"},
        # Meaning: If more than 80% of your input data has issues (like missing values, outliers, or wrong format), it's a problem.
        "model_drift": {"threshold": 0.2, "severity": "high"}
        # Meaning: If more than 80% of your input data has issues (like missing values, outliers, or wrong format), it's a problem.
        # You use metrics like Population Stability Index (PSI) or KL Divergence to check drift.
        # If the drift score > 0.2 → it means the model might not generalize well anymore.
    }
    
}

''' How to calculate forecast error / model accuracy
Let’s say:

Your model predicts 300 units for next week.

The actual number of units sold next week is 450.

➤ Calculate Forecast Error:
The formula:

Forecast Error=∣Actual−Predicted∣
Actual
Forecast Error= 
Actual
∣Actual−Predicted∣
​
 
=
∣450−300∣/450
=
150/450
=
0.333
=
33.3%

 Since 33.3% > 30% threshold, this triggers high_forecast_error'''

ENVIRONMENT = BASE_ENVIRONMENT  # Inherit from base config

if ENVIRONMENT == "production":
    # Production overrides
    LOGGING_CONFIG["level"] = "WARNING"
    PERFORMANCE_CONFIG["max_workers"] = 8
    CACHE_CONFIG["forecast_cache_ttl"] = 7200  # 2 hours
    
elif ENVIRONMENT == "development":
    # Development overrides
    LOGGING_CONFIG["level"] = "DEBUG"
    PERFORMANCE_CONFIG["max_workers"] = 2
    CACHE_CONFIG["forecast_cache_ttl"] = 600  # 10 minutes

elif ENVIRONMENT == "testing":
    # Testing overrides - use different database name
    DATABASE_NAME = "warehouse_management_test"
    CACHE_CONFIG["forecast_cache_ttl"] = 60  # 1 minute
    PERFORMANCE_CONFIG["max_workers"] = 1


# Model Caching Strategy
MODEL_CACHE_STRATEGY = os.getenv("MODEL_CACHE_STRATEGY", "data_hash")  # Options: "never", "time_based", "data_hash", "always"
MODEL_CACHE_HOURS = int(os.getenv("MODEL_CACHE_HOURS", "24"))  # Only used if strategy is "time_based"

# Model Retraining Configuration
# AUTO_RETRAIN_ENABLED = os.getenv("AUTO_RETRAIN_ENABLED", "false").lower() == "true"
# RETRAIN_SCHEDULE_CRON = os.getenv("RETRAIN_SCHEDULE_CRON", "0 2 * * *")  # Daily at 2 AM


def get_kaggle_config() -> Dict[str, Any]:
    """Get Kaggle API configuration."""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    return {
        "username": kaggle_username,
        "key": kaggle_key
    }

def get_prophet_config() -> Dict[str, Any]:
    """Get Prophet model configuration."""
    return PROPHET_CONFIG

def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    return {
        "host": API_HOST,
        "port": API_PORT,
        "prefix": API_PREFIX,
        "title": API_TITLE
    }

def validate_ai_config() -> bool:
    """Validate AI-specific configuration settings."""
    # Base validation is handled by base config
    # Add AI-specific validations here if needed
    return True

# Validate configuration on import
validate_ai_config()
