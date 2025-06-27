import os
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# BASIC CONFIGURATION
# =============================================================================

# Project Information
PROJECT_NAME = "Seasonal Inventory Prediction"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "AI-powered seasonal inventory forecasting using Facebook Prophet"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8003"))
API_PREFIX = "/api/v1"
API_TITLE = "Seasonal Inventory API"

# Database Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "warehouse_management")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# WMS Integration
WMS_API_BASE_URL = os.getenv("WMS_API_BASE_URL", "http://localhost:8000/api/v1")
WMS_API_TIMEOUT = 30

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================

# Kaggle API
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# OpenAI API (for additional AI features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Weather API
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5"

# Economic Data APIs
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Federal Reserve Economic Data
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Stock market data

# =============================================================================
# DATA SOURCES CONFIGURATION
# =============================================================================

# Kaggle Datasets Configuration
KAGGLE_DATASETS = {
    "high_priority": [
        {
            "name": "carrie1/ecommerce-data",
            "description": "E-commerce transaction data with seasonal patterns",
            "target_file": "data.csv",
            "date_column": "InvoiceDate",
            "quantity_column": "Quantity",
            "product_column": "StockCode"
        },
        {
            "name": "mkechinov/ecommerce-behavior-data",
            "description": "E-commerce behavior data",
            "target_file": "2019-Oct.csv",
            "date_column": "event_time",
            "quantity_column": "price",
            "product_column": "product_id"
        },
        {
            "name": "olistbr/brazilian-ecommerce",
            "description": "Brazilian e-commerce dataset",
            "target_file": "olist_orders_dataset.csv",
            "date_column": "order_purchase_timestamp",
            "quantity_column": "order_item_id",
            "product_column": "product_id"
        }
    ],
    "medium_priority": [
        {
            "name": "shashwatwork/dataco-smart-supply-chain",
            "description": "Supply chain dataset",
            "target_file": "DataCoSupplyChainDataset.csv",
            "date_column": "order date (DateOrders)",
            "quantity_column": "Order Item Quantity",
            "product_column": "Product Name"
        },
        {
            "name": "prasad22/retail-transactions-dataset",
            "description": "Retail transactions",
            "target_file": "Retail_Data_Transactions.csv",
            "date_column": "Transaction_Date",
            "quantity_column": "Quantity",
            "product_column": "Product_Category"
        }
    ]
}

# Data Paths
DATA_DIR = "data"
DATASETS_DIR = f"{DATA_DIR}/datasets"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = f"{DATA_DIR}/models"
CACHE_DIR = f"{DATA_DIR}/cache"

# =============================================================================
# PROPHET MODEL CONFIGURATION
# =============================================================================

# Default Prophet Parameters
PROPHET_CONFIG = {
    "base_model": {
        "growth": "linear",
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "mcmc_samples": 0,
        "interval_width": 0.8,
        "uncertainty_samples": 1000
    },
    
    # Custom Seasonalities
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
    
    # Holidays Configuration
    "holidays": {
        "countries": ["US", "BR", "GB", "IN"],
        "custom_events": [
            {"holiday": "Black Friday", "date": "2024-11-29"},
            {"holiday": "Cyber Monday", "date": "2024-12-02"},
            {"holiday": "Back to School", "date": "2024-08-15"},
            {"holiday": "Valentine's Day", "date": "2024-02-14"},
            {"holiday": "Mother's Day", "date": "2024-05-12"},
            {"holiday": "Father's Day", "date": "2024-06-16"}
        ]
    }
}

# Model Training Configuration
TRAINING_CONFIG = {
    "cross_validation": {
        "initial": "730 days",  # 2 years
        "period": "90 days",    # Every 3 months
        "horizon": "90 days"    # 3 months ahead
    },
    
    "hyperparameter_tuning": {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0]
    },
    
    "validation": {
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1,
        "min_train_days": 365
    }
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Feature Engineering Settings
FEATURE_CONFIG = {
    "temporal_features": [
        "year", "month", "day", "dayofweek", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
    ],
    
    "lag_features": {
        "periods": [7, 14, 30, 90, 365],
        "stats": ["mean", "std", "min", "max", "median"]
    },
    
    "rolling_features": {
        "windows": [7, 14, 30, 90],
        "stats": ["mean", "std", "trend"]
    },
    
    "external_features": [
        "temperature", "precipitation", "humidity",
        "consumer_price_index", "unemployment_rate",
        "stock_market_index"
    ]
}

# Data Quality Checks
DATA_QUALITY = {
    "min_data_points": 100,
    "max_missing_ratio": 0.3,
    "outlier_threshold": 3.0,  # Standard deviations
    "min_forecast_horizon": 7,
    "max_forecast_horizon": 365
}

# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

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

# =============================================================================
# CACHING AND PERFORMANCE
# =============================================================================

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

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

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

# =============================================================================
# BUSINESS RULES AND THRESHOLDS
# =============================================================================

# Business Configuration
BUSINESS_CONFIG = {
    "inventory_thresholds": {
        "safety_stock_multiplier": 1.5,
        "reorder_point_multiplier": 2.0,
        "max_stock_multiplier": 4.0,
        "stockout_probability_threshold": 0.05
    },
    
    "seasonality_detection": {
        "min_seasonal_strength": 0.3,
        "trend_change_threshold": 0.1,
        "anomaly_threshold": 2.5
    },
    
    "forecast_validation": {
        "max_growth_rate": 2.0,  # 200% growth
        "min_accuracy_score": 0.6,
        "confidence_interval_coverage": 0.8
    }
}

# Alert Configuration
ALERT_CONFIG = {
    "email_notifications": True,
    "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
    "alert_cooldown": 3600,  # 1 hour
    
    "alert_types": {
        "low_accuracy": {"threshold": 0.7, "severity": "medium"},
        "high_forecast_error": {"threshold": 0.3, "severity": "high"},
        "data_quality_issues": {"threshold": 0.8, "severity": "medium"},
        "model_drift": {"threshold": 0.2, "severity": "high"}
    }
}

# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Environment Settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

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
    # Testing overrides
    DATABASE_NAME = "warehouse_management_test"
    CACHE_CONFIG["forecast_cache_ttl"] = 60  # 1 minute
    PERFORMANCE_CONFIG["max_workers"] = 1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_kaggle_config() -> Dict[str, Any]:
    """Get Kaggle API configuration."""
    return {
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }

def get_database_config() -> Dict[str, str]:
    """Get database configuration."""
    return {
        "mongodb_url": MONGODB_URL,
        "database_name": DATABASE_NAME,
        "redis_url": REDIS_URL
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

def validate_config() -> bool:
    """Validate configuration settings."""
    required_vars = [
        "MONGODB_URL",
        "DATABASE_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return True

# Validate configuration on import
validate_config()
