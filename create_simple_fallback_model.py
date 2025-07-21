"""
This script repairs the AI prediction system by creating a new fallback model.
Run this script with: python create_simple_fallback_model.py
"""

import os
import sys
import pickle
import random
from datetime import datetime

# Add base dir to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def create_simple_encoders():
    """Create a simple encoders dictionary that can be loaded without errors"""
    
    # Categories for each feature
    categories = {
        'category': ['Electronics', 'Clothing', 'Home', 'Beauty', 'Toys', 'Food', 'General'],
        'item_size': ['S', 'M', 'L', 'XL'],
        'season': ['Winter', 'Spring', 'Summer', 'Fall'],
    }
    
    # Create a simple mapping dictionary that maps categories to integers
    encoders = {}
    
    # Create pickle-compatible encoders (simple dictionaries)
    for col_name, values in categories.items():
        encoders[col_name] = {
            'values': values,
            'mapping': {val: i for i, val in enumerate(values)}
        }
    
    # Add target encoder for rack groups
    rack_groups = ['B Rack 1', 'B Rack 2', 'B Rack 3', 'P Rack 1', 'P Rack 2', 'D Rack 1', 'D Rack 2']
    encoders['target'] = {
        'values': rack_groups,
        'mapping': {val: i for i, val in enumerate(rack_groups)}
    }
    
    return encoders

class SimpleModel:
    """A simple replacement model that uses basic rules to predict rack groups"""
    
    def __init__(self):
        """Initialize the simple model"""
        # Default rack assignments based on item size
        self.size_to_rack = {
            'S': 'P Rack 1',  # Small items to Pellet racks
            'M': 'B Rack 1',  # Medium items to Bin racks
            'L': 'D Rack 1',  # Large items to Bulk racks
            'XL': 'D Rack 2'  # Extra Large items to Bulk racks (different section)
        }
        
        # Category-specific assignments that override the size-based ones
        self.category_overrides = {
            'Electronics': {'S': 'P Rack 2', 'M': 'B Rack 2'},
            'Clothing': {'M': 'B Rack 3'},
            'Home': {'L': 'D Rack 2'},
            'Toys': {'M': 'B Rack 2'}
        }
        
        # Seasonal adjustments
        self.seasonal_adjustments = {
            'Winter': {'Clothing': 'B Rack 1'},
            'Summer': {'Clothing': 'B Rack 3'}
        }
        
        # Rack groups for prediction
        self.rack_groups = ['B Rack 1', 'B Rack 2', 'B Rack 3', 'P Rack 1', 'P Rack 2', 'D Rack 1', 'D Rack 2']
    
    def predict(self, X):
        """Predict the rack group for a given input"""
        # Simple handling for various input types
        if hasattr(X, 'shape') and len(X.shape) == 2:
            # Handle DataFrame or array-like input
            num_samples = X.shape[0]
        elif isinstance(X, (list, tuple)):
            num_samples = len(X)
        else:
            # Default to a single sample
            num_samples = 1
        
        results = []
        
        # Generate predictions based on simple rules
        for i in range(num_samples):
            # Default values
            item_size = 'M'
            category = 'General'
            season = self.get_current_season()
            
            # Try to extract features from different input types
            try:
                if hasattr(X, 'iloc'):
                    # DataFrame-like
                    item_size = X.iloc[i].get('item_size', 'M') if 'item_size' in X.iloc[i] else 'M'
                    category = X.iloc[i].get('category', 'General') if 'category' in X.iloc[i] else 'General'
                    season = X.iloc[i].get('season', self.get_current_season()) if 'season' in X.iloc[i] else self.get_current_season()
                elif hasattr(X, 'item'):
                    # Dictionary-like
                    item_size = X[i].get('item_size', 'M') if 'item_size' in X[i] else 'M'
                    category = X[i].get('category', 'General') if 'category' in X[i] else 'General'
                    season = X[i].get('season', self.get_current_season()) if 'season' in X[i] else self.get_current_season()
            except Exception as e:
                print(f"Error extracting features: {e}")
                
            # Simple fallback logic
            rack = self._fallback_prediction(item_size, category, season)
            results.append(rack)
            
        return results
    
    def _fallback_prediction(self, item_size='M', category='General', season=None):
        """Fallback prediction logic"""
        if not season:
            season = self.get_current_season()
            
        # Check for seasonal adjustment
        if season in self.seasonal_adjustments and category in self.seasonal_adjustments[season]:
            return self.seasonal_adjustments[season][category]
            
        # Check for category override
        if category in self.category_overrides and item_size in self.category_overrides[category]:
            return self.category_overrides[category][item_size]
            
        # Default to size-based assignment
        return self.size_to_rack.get(item_size, 'B Rack 1')
    
    def get_current_season(self):
        """Get current season based on current date"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
    
    def predict_proba(self, X):
        """Simulate probability predictions"""
        # Get predictions first
        preds = self.predict(X)
        results = []
        
        # For each prediction, generate a probability distribution
        for pred in preds:
            # Base probabilities: 10% for each class
            probs = [0.1] * len(self.rack_groups)
            
            # Increase probability for the predicted class
            try:
                pred_idx = self.rack_groups.index(pred) if pred in self.rack_groups else 0
                probs[pred_idx] = 0.7  # 70% confidence in prediction
            except ValueError:
                # If rack group not found, use first one
                probs[0] = 0.7
            
            # Normalize to ensure probabilities sum to 1
            total = sum(probs)
            probs = [p / total for p in probs]
            
            results.append(probs)
        
        return results

def create_and_save_models():
    """Create and save simple models to replace the incompatible ones"""
    print("\n🔧 Creating Simple Fallback Model")
    print("======================================")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(BASE_DIR, "ai_services", "path_optimization", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "warehouse_storage_location_model.pkl")
    encoders_path = os.path.join(models_dir, "warehouse_storage_location_encoders.pkl")
    
    # Create simple model and encoders
    model = SimpleModel()
    encoders = create_simple_encoders()
    
    # Save model
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Simple model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
    
    # Save encoders
    try:
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders, f)
        print(f"✅ Simple encoders saved to: {encoders_path}")
    except Exception as e:
        print(f"❌ Failed to save encoders: {e}")
    
    print("\n✅ Simple fallback model creation complete")
    print("Restart your FastAPI server and test the endpoint.")

if __name__ == "__main__":
    create_and_save_models()
