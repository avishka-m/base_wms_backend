#demand predcition for next day
#need to run this daily 
import pandas as pd
from datetime import datetime
from pathlib import Path
from .prophet_forecaster import ProphetForecaster
from base_wms_backend.app.services.inventory_service import InventoryService
from ..config import PROCESSED_DIR

# Load processed demand data
wms_file = Path(PROCESSED_DIR) / "daily_demand_by_product_modern.csv"
data = pd.read_csv(wms_file)

# Get unique products
product_ids = data['product_id'].unique()

today = datetime.today().date()
periods = 1  # Forecast for the next day

for product_id in product_ids:
    try:
        forecaster = ProphetForecaster(product_id=str(product_id))
        forecaster.train(data)
        forecast = forecaster.predict(periods=periods, include_history=False)
        for _, row in forecast.iterrows():
            InventoryService.save_demand_forecast(
                product_id=str(product_id),
                date=row['ds'],
                predicted_demand=row['yhat']
            )
        print(f"Saved daily forecast for product {product_id}")
    except Exception as e:
        print(f"Error forecasting for product {product_id}: {e}")

print("daily forecasts updated for all products.")
