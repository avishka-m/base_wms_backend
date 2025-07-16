"""
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
