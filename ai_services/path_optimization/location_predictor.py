# ai_services/path_optimization/location_predictor.py

import pickle
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not installed, using fallback predictions")
    SKLEARN_AVAILABLE = False


from .warehouse_mapper import warehouse_mapper

class LocationPredictor:
    """Handles location prediction using trained ML model"""
    
    def __init__(self, model_path: str = None, encoders_path: str = None):
        self.model = None
        self.encoders = None
        
        # Set default paths for both files
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.model_path = model_path or os.path.join(models_dir, 'warehouse_storage_location_model.pkl')
        self.encoders_path = encoders_path or os.path.join(models_dir, 'warehouse_storage_location_encoders.pkl')
        
        self.load_model()
        self.load_encoders()
    
    
    def load_model(self):
        """Load the trained model from pickle file with improved error handling"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"⚠️ Model file not found at {self.model_path}")
            print("   Using fallback prediction logic")
            self.model = None
        except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError, ValueError) as e:
            print(f"⚠️ Error loading model (may be from different Python/sklearn version): {str(e)}")
            print("   Using fallback prediction logic")
            self.model = None
        except Exception as e:
            print(f"⚠️ Unexpected error loading model: {str(e)}")
            self.model = None
    
    def load_encoders(self):
        """Load the encoders from pickle file"""
        try:
            with open(self.encoders_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print(f"Encoders loaded successfully from {self.encoders_path}")
        except FileNotFoundError:
            print(f"Encoders file not found at {self.encoders_path}")
            self.encoders = None
        except Exception as e:
            print(f"Error loading encoders: {str(e)}")
            self.encoders = None
    
    def get_current_season(self) -> str:
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
    
    async def get_seasonal_score(self, item_id: int, db_collection) -> float:
        """Get seasonal score from seasonal_demand collection"""
        try:
            # Query seasonal_demand collection for the item
            seasonal_data = db_collection.find_one({
                "itemID": item_id,
                "season": self.get_current_season()
            })
            
            if seasonal_data:
                return float(seasonal_data.get('seasonal_score', 0.5))
            else:
                # Default seasonal score if no data found
                return 0.5
                
        except Exception as e:
            print(f"Error getting seasonal score: {str(e)}")
            return 0.5  # Default value
    
    def prepare_model_input(self, 
                           category: str, 
                           item_size: str, 
                           season: str, 
                           seasonal_score: float) -> pd.DataFrame:
        """Prepare input data for the model with proper encoding"""
        
        # Create input dataframe with the expected features
        input_data = pd.DataFrame({
            'category': [category],
            'item_size': [item_size],
            'season': [season],
            'seasonal_score': [seasonal_score]
        })
        
        # Apply encoders if available
        if self.encoders:
            try:
                # Apply encoders to categorical columns
                for column, encoder in self.encoders.items():
                    if column in input_data.columns:
                        # Handle unseen categories gracefully
                        try:
                            input_data[column] = encoder.transform(input_data[column])
                        except ValueError as e:
                            print(f"Warning: Unseen category in {column}: {input_data[column].iloc[0]}")
                            # Use the most frequent category or a default
                            if hasattr(encoder, 'classes_'):
                                input_data[column] = encoder.transform([encoder.classes_[0]])
                            else:
                                # For LabelEncoder, use 0 as default
                                input_data[column] = [0]
            except Exception as e:
                print(f"Error applying encoders: {str(e)}")
                # Continue with raw data if encoding fails
        
        return input_data
    
    def predict_rack_group(self, 
                          category: str, 
                          item_size: str, 
                          seasonal_score: float) -> str:
        """Predict rack group using the trained model"""
        
        if self.model is None:
            # Fallback logic if model is not available
            return self._fallback_prediction(item_size)
        
        try:
            season = self.get_current_season()
            input_data = self.prepare_model_input(category, item_size, season, seasonal_score)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Handle encoded predictions (if prediction is numeric, decode it)
            if isinstance(prediction, (int, float)) and self.encoders and 'target' in self.encoders:
                try:
                    # Decode the prediction back to string
                    target_encoder = self.encoders['target']
                    if hasattr(target_encoder, 'inverse_transform'):
                        prediction = target_encoder.inverse_transform([int(prediction)])[0]
                except Exception as e:
                    print(f"Error decoding prediction: {str(e)}")
                    return self._fallback_prediction(item_size)
            
            # Validate prediction
            valid_rack_groups = [
                'B Rack 1', 'B Rack 2', 'B Rack 3',
                'P Rack 1', 'P Rack 2',
                'D Rack 1', 'D Rack 2'
            ]
            
            if prediction in valid_rack_groups:
                return prediction
            else:
                print(f"Invalid prediction: {prediction}, using fallback")
                return self._fallback_prediction(item_size)
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return self._fallback_prediction(item_size)
    
    def _fallback_prediction(self, item_size: str) -> str:
        """Fallback prediction logic based on item size"""
        item_size = item_size.upper()
        
        if item_size == 'S':
            return 'P Rack 1'  # Small items to Pallet racks
        elif item_size == 'L':
            return 'D Rack 1'  # Large items to Bulk racks
        elif item_size == 'XL':
            return 'D Rack 2'  # Extra Large items to Bulk racks (different section)
        else:  # Medium or unknown
            return 'B Rack 1'  # Medium items to Bin racks
    
    def get_model_confidence(self, 
                           category: str, 
                           item_size: str, 
                           seasonal_score: float) -> float:
        """Get prediction confidence if model supports it"""
        
        if self.model is None or self.encoders is None:
            return 0.5  # Low confidence for fallback
        
        try:
            # Check if model has predict_proba method
            if hasattr(self.model, 'predict_proba'):
                season = self.get_current_season()
                input_data = self.prepare_model_input(category, item_size, season, seasonal_score)
                probabilities = self.model.predict_proba(input_data)[0]
                return float(max(probabilities))
            else:
                return 0.8  # Default confidence for models without probability
                
        except Exception as e:
            print(f"Error getting confidence: {str(e)}")
            return 0.5
    
    async def predict_optimal_location(self, 
                                     item_id: int,
                                     category: str,
                                     item_size: str,
                                     db_collection_seasonal,
                                     quantity: int = 1) -> Dict[str, Any]:
        """Main method to predict optimal location for an item"""
        
        # Get seasonal score
        seasonal_score = await self.get_seasonal_score(item_id, db_collection_seasonal)
        
        # Predict rack group
        predicted_rack_group = self.predict_rack_group(category, item_size, seasonal_score)
        
        # Get confidence
        confidence = self.get_model_confidence(category, item_size, seasonal_score)
        
        return {
            'rack_group': predicted_rack_group,
            'available_locations': warehouse_mapper.get_rack_locations(predicted_rack_group),
            'confidence': confidence,
            'season': self.get_current_season(),
            'seasonal_score': seasonal_score,
            'quantity': quantity
        }

# Global instance
location_predictor = LocationPredictor()