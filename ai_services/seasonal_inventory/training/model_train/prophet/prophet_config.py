"""
Production Prophet Configuration
Auto-generated optimized hyperparameters per category
Generated on: 2025-01-15
"""

# Optimized hyperparameters for each category
CATEGORY_HYPERPARAMETERS = {
    "books_media": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    "clothing": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    "electronics": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    "health_beauty": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    "home_garden": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "multiplicative",
    },
    "sports_outdoors": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "multiplicative",
    },
}

# Default fallback parameters (if category not found)
DEFAULT_PARAMETERS = {
    "changepoint_prior_scale": 0.5,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "holidays_prior_scale": 10.0
}

# Performance metrics for reference (RMSE values from optimization)
CATEGORY_PERFORMANCE = {
    "books_media": {"rmse": 586.11, "improvement": "29% better than default"},
    "clothing": {"rmse": 3024.42, "improvement": "12% better than default"},  
    "electronics": {"rmse": 2219.41, "improvement": "8% better than default"},
    "health_beauty": {"rmse": 1630.39, "improvement": "22% better than default"},
    "home_garden": {"rmse": 1104.34, "improvement": "18% better than default"},
    "sports_outdoors": {"rmse": 955.39, "improvement": "15% better than default"}
}

def get_category_params(category):
    """Get optimized parameters for a specific category"""
    return CATEGORY_HYPERPARAMETERS.get(category, DEFAULT_PARAMETERS)

def get_all_categories():
    """Get list of all available categories"""
    return list(CATEGORY_HYPERPARAMETERS.keys())

def validate_params(params):
    """Validate parameter values"""
    required_keys = ['changepoint_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']
    return all(key in params for key in required_keys)

# Usage example:
"""
from prophet import Prophet
import prophet_config

# Get optimized parameters for a category
category = 'books_media'
params = prophet_config.get_category_params(category)

# Create and train Prophet model
model = Prophet(**params)
model.fit(training_data)

# Generate forecast
forecast = model.predict(future_dataframe)
"""
