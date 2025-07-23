import pandas as pd
import pickle
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from prophet import Prophet

 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
current_dir = Path(__file__).resolve().parent

# Add this here:
CATEGORY_REGRESSORS = {
    "books_media": ["is_weekend", "month_end_effect", "payday_effect","summer_surge"],
    "clothing": ["is_weekend"],
    "electronics": ["is_weekend","month_end_effect", "payday_effect"],
    "health_beauty": ["is_weekend"],
    "home_garden": ["is_weekend", "month_end_effect", "payday_effect", "summer_surge"],
    "sports_outdoors": ["is_weekend", "month_end_effect", "payday_effect"],
    
}
class ProphetCategoryPredictor:
    """
    Minimal predictor for category-level demand forecasting using pre-trained Prophet models.
    """
    def __init__(self, category: str):
        self.category = category
        self.model = None
        # self.models_dir = Path(MODELS_DIR)
        self.model_path = current_dir.parents[1] / "data" / "models" / f"category_{category}_prophet_model.pkl"
        # self.model_path = self.models_dir / f"category_{category}_prophet_model.pkl"
        logger.info(f"ProphetCategoryPredictor initialized for category: {category}")

    def load_model(self) -> bool:
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False
        with open(self.model_path, "rb") as f:
            # model_data = pickle.load(f)
            # self.model = model_data["model"]
             self.model = pickle.load(f)
        logger.info(f"Loaded model for category: {self.category}")
        return True

    def predict(self, future_dates: List,confidence_interval:float=0.8) -> Optional[pd.DataFrame]:
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        self.model.interval_width = confidence_interval 
        future_df = pd.DataFrame({"ds": pd.to_datetime(future_dates)})
            # Add regressors as used in training
        
        # Dynamically add regressors based on category
        regressors = CATEGORY_REGRESSORS.get(self.category, [])
        if "is_weekend" in regressors:
            future_df["is_weekend"] = future_df["ds"].dt.dayofweek >= 5  # True for Saturday/Sunday
        if "month_end_effect" in regressors:
            future_df["month_end_effect"] = future_df["ds"].dt.is_month_end     
        if "payday_effect" in regressors:
            future_df["payday_effect"] = future_df["ds"].dt.day.isin([1, 15])
      
        if "summer_surge" in regressors:
            # You can adjust this logic as needed. For now, treat it the same as summer_effect.
            future_df["summer_surge"] = future_df["ds"].dt.month.isin([6, 7, 8])
        
        forecast = self.model.predict(future_df)
        forecast["category"] = self.category
        forecast["forecast_date"] = datetime.now()
        logger.info(f"Forecast generated for category {self.category}: {len(forecast)} predictions")
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "category", "forecast_date"]]

    # def plot_forecast(self, forecast: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> None:
    #     import matplotlib.pyplot as plt
    #     if self.model is None:
    #         logger.error("Model not loaded. Call load_model() first.")
    #         return
    #     if forecast is None:
    #         logger.error("No forecast provided for plotting.")
    #         return
    #     fig = self.model.plot(forecast, figsize=(12, 8))
    #     plt.title(f'Seasonal Inventory Forecast - Category: {self.category}')
    #     plt.xlabel('Date')
    #     plt.ylabel('Demand')
    #     handles, labels = fig.gca().get_legend_handles_labels()
    #     custom_labels = [
    #         'Forecast (yhat)',
    #         'Uncertainty Interval',
    #         'Actuals'
    #     ]
    #     if len(labels) >= 3:
    #         plt.legend(handles[:3], custom_labels, loc='upper left')
    #     else:
    #         plt.legend(loc='upper left')
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         logger.info(f"Forecast plot saved to: {save_path}")
    #     plt.show()

# import pickle
# from pathlib import Path

# category="electronics"
# current_dir = Path(__file__).resolve().parent
# model_path = current_dir.parents[1] / "data" / "models" / f"category_{category}_prophet_model.pkl"

# print("Looking for model at:", model_path)
 
# if model_path.exists():
#     with open(model_path, "rb") as f:
#         try:
#             model_obj = pickle.load(f)
#             print(f"Model for category '{category}' loaded successfully!")
#             print(f"Type of loaded object: {type(model_obj)}")
#         except Exception as e:
#             print(f"Error loading model: {e}")
            
# else:
#     print(f"Model file not found: {model_path}")            

if __name__ == "__main__":
    
    # Example usage for testing
    category = "home_garden"  # Change to your category
    predictor = ProphetCategoryPredictor(category)
    if predictor.load_model():
        # Generate a list of future dates (e.g., next 7 days)
        from datetime import timedelta
        today = datetime.today()
        future_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        forecast = predictor.predict(future_dates, confidence_interval=0.8)
        if forecast is not None:
            print(forecast)
            predictor.plot_forecast(forecast)
        else:
            print("Prediction failed.")
    else:
        print("Model loading failed.")