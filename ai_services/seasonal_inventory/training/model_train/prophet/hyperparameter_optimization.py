"""
Hyperparameter Optimization for Each Category
Find optimal Prophet parameters per category and create production config system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import yaml
import warnings
from itertools import product
import time

warnings.filterwarnings('ignore')

def optimize_category_hyperparameters():
    """
    Find optimal hyperparameter    print(f"\n🚀 READY FOR PRODUCTION!")
    print("  1. Use config_loader.py to load parameters")
    print("  2. Follow production_example.py for implementation")
    print("  3. Deploy with environment-specific configs")r each category using grid search
    """
    
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION BY CATEGORY")
    print("="*80)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    categories = df['category'].unique()
    print(f"Categories to optimize: {list(categories)}")
    
    # Define hyperparameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 50.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'daily_seasonality': [False],  # Keep False for daily data
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }
    
    print(f"Parameter grid size: {len(list(product(*param_grid.values())))} combinations")
    print("Running grid search for each category...")
    
    optimal_params = {}
    category_results = {}
    
    for category in categories:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING: {category.upper()}")
        print(f"{'='*60}")
        
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Data points: {len(cat_data)}")
        print(f"Date range: {cat_data['ds'].min()} to {cat_data['ds'].max()}")
        
        # Split data (80% train, 20% validation)
        split_idx = int(len(cat_data) * 0.8)
        train_data = cat_data[:split_idx].copy()
        val_data = cat_data[split_idx:].copy()
        
        print(f"Train: {len(train_data)} days, Validation: {len(val_data)} days")
        
        # Grid search with reduced combinations for efficiency
        best_score = float('inf')
        best_params = None
        results = []
        
        # Simplified grid for efficiency (top performing combinations)
        simplified_grid = [
            # Conservative approaches
            {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            
            # Balanced approaches  
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'multiplicative'},
            
            # Flexible approaches
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
            
            # High flexibility
            {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
            {'changepoint_prior_scale': 1.0, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
            
            # Multiplicative variants
            {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
            {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'multiplicative'},
        ]
        
        print(f"Testing {len(simplified_grid)} parameter combinations...")
        
        for i, params in enumerate(simplified_grid):
            try:
                # Add default values
                full_params = {
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False,
                    'holidays_prior_scale': 10.0,
                    **params
                }
                
                # Train model
                model = Prophet(**full_params)
                model.fit(train_data)
                
                # Predict on validation set
                future = model.make_future_dataframe(periods=len(val_data))
                forecast = model.predict(future)
                
                # Calculate validation metrics
                val_pred = forecast['yhat'][split_idx:].values
                val_actual = val_data['y'].values
                
                rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
                mae = mean_absolute_error(val_actual, val_pred)
                r2 = r2_score(val_actual, val_pred)
                mape = np.mean(np.abs((val_actual - val_pred) / val_actual)) * 100
                
                results.append({
                    'params': full_params,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape
                })
                
                if rmse < best_score:
                    best_score = rmse
                    best_params = full_params.copy()
                
                print(f"  [{i+1:2d}/{len(simplified_grid)}] RMSE: {rmse:7.2f} | {params}")
                
            except Exception as e:
                print(f"  [{i+1:2d}/{len(simplified_grid)}] ERROR: {str(e)[:50]}...")
                continue
        
        optimal_params[category] = best_params
        category_results[category] = {
            'best_params': best_params,
            'best_rmse': best_score,
            'all_results': results
        }
        
        print(f"\n🏆 BEST PARAMETERS FOR {category}:")
        print(f"   RMSE: {best_score:.2f}")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
    
    return optimal_params, category_results

def create_config_files(optimal_params):
    """
    Create configuration files for production use
    """
    
    print(f"\n{'='*60}")
    print("CREATING CONFIGURATION FILES")
    print(f"{'='*60}")
    
    # 1. JSON Configuration
    json_config = {
        "model_config": {
            "prophet_version": "1.1.0",
            "created_date": "2025-01-15",
            "description": "Optimized Prophet hyperparameters per category",
            "validation_method": "80/20 train-validation split",
            "optimization_metric": "RMSE"
        },
        "categories": {}
    }
    
    for category, params in optimal_params.items():
        json_config["categories"][category] = {
            "hyperparameters": params,
            "notes": f"Optimized for {category} category demand forecasting"
        }
    
    # Save JSON config
    with open('prophet_config.json', 'w') as f:
        json.dump(json_config, f, indent=4)
    
    # 2. YAML Configuration (more human-readable)
    yaml_config = {
        'model_config': {
            'prophet_version': '1.1.0',
            'created_date': '2025-01-15',
            'description': 'Optimized Prophet hyperparameters per category',
            'validation_method': '80/20 train-validation split',
            'optimization_metric': 'RMSE'
        },
        'categories': {}
    }
    
    for category, params in optimal_params.items():
        yaml_config['categories'][category] = {
            'hyperparameters': params,
            'notes': f'Optimized for {category} category demand forecasting'
        }
    
    # Save YAML config
    with open('prophet_config.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, indent=2)
    
    # 3. Python Configuration Module
    python_config = f'''"""
Production Prophet Configuration
Auto-generated optimized hyperparameters per category
Generated on: 2025-01-15
"""

# Optimized hyperparameters for each category
CATEGORY_HYPERPARAMETERS = {{
'''
    
    for category, params in optimal_params.items():
        python_config += f'    "{category}": {{\n'
        for key, value in params.items():
            if isinstance(value, str):
                python_config += f'        "{key}": "{value}",\n'
            else:
                python_config += f'        "{key}": {value},\n'
        python_config += f'    }},\n'
    
    python_config += '''}

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
'''
    
    # Save Python config
    with open('prophet_config.py', 'w') as f:
        f.write(python_config)
    
    # 4. Environment-specific configs (dev, staging, prod)
    environments = ['development', 'staging', 'production']
    
    for env in environments:
        env_config = {
            'environment': env,
            'model_config': json_config['model_config'].copy(),
            'categories': json_config['categories'].copy()
        }
        
        # Adjust for environment
        if env == 'development':
            env_config['model_config']['description'] += ' - Development environment'
            # Maybe use faster/less accurate params for dev
        elif env == 'staging':
            env_config['model_config']['description'] += ' - Staging environment'
        else:  # production
            env_config['model_config']['description'] += ' - Production environment'
        
        with open(f'prophet_config_{env}.json', 'w') as f:
            json.dump(env_config, f, indent=4)
    
    print("✅ Configuration files created:")
    print("   • prophet_config.json (JSON format)")
    print("   • prophet_config.yaml (YAML format)")
    print("   • prophet_config.py (Python module)")
    print("   • prophet_config_development.json")
    print("   • prophet_config_staging.json")
    print("   • prophet_config_production.json")

def create_config_loader():
    """
    Create a configuration loader utility
    """
    
    config_loader_code = '''"""
Configuration Loader for Prophet Models
Handles loading and validation of hyperparameters
"""

import json
import yaml
import os
from typing import Dict, Any, Optional

class ProphetConfigLoader:
    """
    Utility class for loading Prophet configuration
    """
    
    def __init__(self, config_path: str = None, environment: str = 'production'):
        self.environment = environment
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default config path based on environment"""
        return f"prophet_config_{self.environment}.json"
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        file_ext = os.path.splitext(self.config_path)[1].lower()
        
        with open(self.config_path, 'r') as f:
            if file_ext == '.json':
                return json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {file_ext}")
    
    def get_category_params(self, category: str) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific category
        
        Args:
            category: Category name (e.g., 'books_media')
            
        Returns:
            Dictionary of hyperparameters
        """
        if category not in self.config['categories']:
            available = list(self.config['categories'].keys())
            raise ValueError(f"Category '{category}' not found. Available: {available}")
        
        return self.config['categories'][category]['hyperparameters']
    
    def get_all_categories(self) -> list:
        """Get list of all available categories"""
        return list(self.config['categories'].keys())
    
    def validate_category(self, category: str) -> bool:
        """Check if category exists in config"""
        return category in self.config['categories']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration metadata"""
        return self.config.get('model_config', {})
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()

# Usage examples:
if __name__ == "__main__":
    # Load production config
    config = ProphetConfigLoader(environment='production')
    
    # Get parameters for specific category
    books_params = config.get_category_params('books_media')
    print("Books & Media parameters:", books_params)
    
    # Get all available categories
    categories = config.get_all_categories()
    print("Available categories:", categories)
    
    # Get model info
    info = config.get_model_info()
    print("Model info:", info)
'''
    
    with open('config_loader.py', 'w') as f:
        f.write(config_loader_code)
    
    print("✅ Config loader utility created: config_loader.py")

def create_production_example():
    """
    Create example of how to use configs in production
    """
    
    example_code = '''"""
Production Example: Using Prophet with Optimized Hyperparameters
"""

from prophet import Prophet
import pandas as pd
from config_loader import ProphetConfigLoader

class OptimizedProphetForecaster:
    """
    Production forecaster using optimized hyperparameters per category
    """
    
    def __init__(self, environment='production'):
        self.config_loader = ProphetConfigLoader(environment=environment)
        self.models = {}
    
    def train_category_model(self, category: str, data: pd.DataFrame):
        """
        Train Prophet model for specific category using optimized parameters
        
        Args:
            category: Category name
            data: DataFrame with 'ds' and 'y' columns
        """
        # Get optimized parameters for this category
        params = self.config_loader.get_category_params(category)
        
        print(f"Training {category} model with optimized parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Create and train model
        model = Prophet(**params)
        model.fit(data)
        
        self.models[category] = model
        return model
    
    def predict(self, category: str, periods: int = 30):
        """
        Generate forecast for specific category
        
        Args:
            category: Category name
            periods: Number of periods to forecast
            
        Returns:
            Forecast DataFrame
        """
        if category not in self.models:
            raise ValueError(f"Model for category '{category}' not trained")
        
        model = self.models[category]
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast
    
    def get_model_params(self, category: str):
        """Get the parameters used for a specific category"""
        return self.config_loader.get_category_params(category)

# Example usage:
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = OptimizedProphetForecaster(environment='production')
    
    # Load your data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Train models for each category with optimized parameters
    for category in forecaster.config_loader.get_all_categories():
        # Prepare category data
        cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        
        # Train with optimized parameters
        forecaster.train_category_model(category, cat_data)
        
        # Generate forecast
        forecast = forecaster.predict(category, periods=30)
        print(f"\\n{category} forecast generated: {len(forecast)} periods")
    
    print("\\n✅ All category models trained with optimized hyperparameters!")
'''
    
    with open('production_example.py', 'w') as f:
        f.write(example_code)
    
    print("✅ Production example created: production_example.py")

if __name__ == "__main__":
    # Run hyperparameter optimization
    print("Starting hyperparameter optimization...")
    start_time = time.time()
    
    optimal_params, category_results = optimize_category_hyperparameters()
    
    end_time = time.time()
    print(f"\\nOptimization completed in {end_time - start_time:.1f} seconds")
    
    # Create configuration files
    create_config_files(optimal_params)
    
    # Create utilities
    create_config_loader()
    create_production_example()
    
    # Summary
    print(f"\\n{'='*80}")
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    
    print("\\n📊 OPTIMAL PARAMETERS BY CATEGORY:")
    for category, params in optimal_params.items():
        rmse = category_results[category]['best_rmse']
        print(f"\\n{category.upper()} (RMSE: {rmse:.2f}):")
        for key, value in params.items():
            print(f"  • {key}: {value}")
    
    print(f"\\n📁 FILES CREATED:")
    print("  Config Files:")
    print("    • prophet_config.json")
    print("    • prophet_config.yaml") 
    print("    • prophet_config.py")
    print("    • prophet_config_[environment].json")
    print("  Utilities:")
    print("    • config_loader.py")
    print("    • production_example.py")
    
    print(f"\\n🚀 READY FOR PRODUCTION!")
    print("  1. Use config_loader.py to load parameters")
    print("  2. Follow production_example.py for implementation")
    print("  3. Deploy with environment-specific configs")
