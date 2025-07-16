"""
Cross-Validation Optimized Prophet Configuration
Comprehensive hyperparameter optimization using Prophet's cross-validation
Generated on: 2025-01-15
"""

# Cross-validation optimized hyperparameters (best performing)
CATEGORY_HYPERPARAMETERS = {
    "books_media": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 10.0,
        "seasonality_mode": "additive",
    },
    "clothing": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 0.1,
        "seasonality_mode": "additive",
    },
    "electronics": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 0.1,
        "seasonality_mode": "additive",
    },
    "health_beauty": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.1,
        "seasonality_prior_scale": 50.0,
        "seasonality_mode": "multiplicative",
    },
    "home_garden": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 0.5,
        "seasonality_prior_scale": 1.0,
        "seasonality_mode": "additive",
    },
    "sports_outdoors": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "holidays_prior_scale": 10.0,
        "changepoint_prior_scale": 1.0,
        "seasonality_prior_scale": 0.1,
        "seasonality_mode": "additive",
    },
}

# Cross-validation performance metrics
CATEGORY_CV_PERFORMANCE = {
    "books_media": {
        "cv_rmse": 509.59,
        "cv_rmse_std": 249.91,
        "cv_mae": 328.99,
        "cv_mape": 0.10,
        "stability_pct": 49.0,
        "cv_folds": 82,
        "rmse_min": 123.24,
        "rmse_max": 918.18,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
    "clothing": {
        "cv_rmse": 2781.04,
        "cv_rmse_std": 1780.33,
        "cv_mae": 1768.66,
        "cv_mape": 0.12,
        "stability_pct": 64.0,
        "cv_folds": 82,
        "rmse_min": 718.70,
        "rmse_max": 7313.86,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
    "electronics": {
        "cv_rmse": 2023.35,
        "cv_rmse_std": 1327.80,
        "cv_mae": 1290.28,
        "cv_mape": 0.14,
        "stability_pct": 65.6,
        "cv_folds": 82,
        "rmse_min": 569.96,
        "rmse_max": 5458.67,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
    "health_beauty": {
        "cv_rmse": 1630.83,
        "cv_rmse_std": 665.88,
        "cv_mae": 1162.32,
        "cv_mape": 0.12,
        "stability_pct": 40.8,
        "cv_folds": 82,
        "rmse_min": 634.50,
        "rmse_max": 2763.80,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
    "home_garden": {
        "cv_rmse": 1008.84,
        "cv_rmse_std": 483.43,
        "cv_mae": 671.07,
        "cv_mape": 0.10,
        "stability_pct": 47.9,
        "cv_folds": 82,
        "rmse_min": 257.40,
        "rmse_max": 1798.16,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
    "sports_outdoors": {
        "cv_rmse": 869.26,
        "cv_rmse_std": 457.56,
        "cv_mae": 559.32,
        "cv_mape": 0.10,
        "stability_pct": 52.6,
        "cv_folds": 82,
        "rmse_min": 198.51,
        "rmse_max": 1624.03,
        "validation_method": "cross_validation",
        "stability_rating": "variable"
    },
}

# Top 3 alternative parameter sets per category
CATEGORY_ALTERNATIVES = {
    "books_media": [
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            "rmse": 509.59,
            "stability": 49.0
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
            "rmse": 509.69,
            "stability": 49.0
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
            "rmse": 511.42,
            "stability": 48.8
        },
    ],
    "clothing": [
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
            "rmse": 2781.04,
            "stability": 64.0
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
            "rmse": 2800.27,
            "stability": 63.6
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            "rmse": 2805.65,
            "stability": 63.4
        },
    ],
    "electronics": [
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
            "rmse": 2023.35,
            "stability": 65.6
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            "rmse": 2024.34,
            "stability": 65.6
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            "rmse": 2024.73,
            "stability": 65.6
        },
    ],
    "health_beauty": [
        {
            "params": {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'multiplicative'},
            "rmse": 1630.83,
            "stability": 40.8
        },
        {
            "params": {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
            "rmse": 1633.46,
            "stability": 40.7
        },
        {
            "params": {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
            "rmse": 1635.00,
            "stability": 40.5
        },
    ],
    "home_garden": [
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            "rmse": 1008.84,
            "stability": 47.9
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
            "rmse": 1009.12,
            "stability": 47.9
        },
        {
            "params": {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
            "rmse": 1017.76,
            "stability": 46.9
        },
    ],
    "sports_outdoors": [
        {
            "params": {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'additive'},
            "rmse": 869.26,
            "stability": 52.6
        },
        {
            "params": {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
            "rmse": 869.30,
            "stability": 51.7
        },
        {
            "params": {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1, 'seasonality_mode': 'multiplicative'},
            "rmse": 869.74,
            "stability": 51.9
        },
    ],
}

# Cross-validation configuration used
CV_CONFIG = {'initial': '600 days', 'period': '120 days', 'horizon': '90 days'}

def get_category_params(category):
    """Get cross-validation optimized parameters for a specific category"""
    if category not in CATEGORY_HYPERPARAMETERS:
        raise ValueError(f"Category '{category}' not found. Available: {list(CATEGORY_HYPERPARAMETERS.keys())}")
    return CATEGORY_HYPERPARAMETERS[category]

def get_category_performance(category):
    """Get cross-validation performance metrics for a category"""
    return CATEGORY_CV_PERFORMANCE.get(category, {})

def get_alternative_params(category, rank=1):
    """Get alternative parameter sets (rank 1=best, 2=second best, etc.)"""
    if category not in CATEGORY_ALTERNATIVES:
        raise ValueError(f"Category '{category}' not found")
    if rank < 1 or rank > len(CATEGORY_ALTERNATIVES[category]):
        raise ValueError(f"Rank must be between 1 and {len(CATEGORY_ALTERNATIVES[category])}")
    
    alt_params = CATEGORY_ALTERNATIVES[category][rank-1]['params']
    # Add default values
    full_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0,
        **alt_params
    }
    return full_params

def get_all_categories():
    """Get list of all available categories"""
    return list(CATEGORY_HYPERPARAMETERS.keys())

def get_stability_rating(category):
    """Get stability rating for a category (excellent/good/variable)"""
    return CATEGORY_CV_PERFORMANCE.get(category, {}).get('stability_rating', 'unknown')

# Default fallback parameters
DEFAULT_PARAMETERS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "seasonality_mode": "additive",
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "holidays_prior_scale": 10.0
}
