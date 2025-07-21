# Endpoint to get available products (for frontend compatibility)


import os
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import pandas as pd
import pickle

from ..services.prophet_forecasting_service import get_prophet_forecasting_service
from ..auth.dependencies import has_role
from ..utils.database import get_collection
from ai_services.seasonal_inventory.config import BUSINESS_CONFIG

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/products")
async def get_available_products(current_user: Dict[str, Any] = Depends(has_role(["Manager"]))):
    """
    Returns a list of products and their categories from the CSV data.
    """
    import csv
    import os
    csv_path = os.path.join(os.path.dirname(__file__), '../../ai_services/seasonal_inventory/data/processed/daily_demand_by_product_modern.csv')
    products = {}
    try:
        with open(os.path.abspath(csv_path), mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                pid = row.get("product_id")
                cat = row.get("category", "Unknown")
                if pid and pid not in products:
                    products[pid] = {"id": pid, "category": cat}
        product_list = list(products.values())
        return {
            "status": "success",
            "data": product_list,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading products from CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read products from CSV: {str(e)}")


# Endpoint to get available categories
@router.get("/categories")
async def get_available_categories(current_user: Dict[str, Any] = Depends(has_role(["Manager"]))):
    """
    Returns a list of unique product categories from the CSV data.
    """
    import csv
    import os
    csv_path = os.path.join(os.path.dirname(__file__), '../../ai_services/seasonal_inventory/data/processed/daily_demand_by_product_modern.csv')
    categories = set()
    try:
        with open(os.path.abspath(csv_path), mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cat = row.get("category")
                if cat:
                    categories.add(cat)
        category_list = sorted(list(categories))
        return {
            "status": "success",
            "data": category_list,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading categories from CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read categories from CSV: {str(e)}")


# Category-level forecast endpoint (category-level model)
@router.get("/category/{category}/predict")
async def get_category_forecast(
    category: str,
    prediction_horizon_days: int = Query(30, ge=1, le=365),
    confidence_interval: float = Query(0.95, ge=0.5, le=0.99),
    include_external_factors: bool = Query(True),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(has_role(["Manager"]))
) -> Dict[str, Any]:
    """
    Get forecast for a category using a category-level Prophet model.
    """
    import os
    import pandas as pd
    from ai_services.seasonal_inventory.src.models.prophet_forecaster import ProphetCategoryPredictor
    from ai_services.seasonal_inventory.config import PROPHET_CONFIG, MODELS_DIR

    # Determine forecast date range
    today = datetime.now().date()
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start_dt = today
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_dt = start_dt + timedelta(days=prediction_horizon_days-1)
    periods = (end_dt - start_dt).days + 1

    # Load category-level model
    #model_config = PROPHET_CONFIG.get(category)
    model_config = PROPHET_CONFIG.get(category)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"No Prophet config found for category '{category}'")

    forecaster = ProphetCategoryPredictor(category=category)

    # Model file path: MODELS_DIR/category_{category}_prophet_model.pkl
    model_path = os.path.join("ai_services/seasonal_inventory/data/models", f"category_{category}_prophet_model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"No trained model found for category '{category}'")

    # ProphetCategoryPredictor loads model from its own path, so just call load_model()
    if not forecaster.load_model():
        raise HTTPException(status_code=500, detail=f"Failed to load model for category '{category}'")

    # Generate forecast

    # ProphetCategoryPredictor expects a list of future dates
    future_dates = [str(start_dt + timedelta(days=i)) for i in range(periods)]
    forecast_df = forecaster.predict(future_dates)
    # Filter to requested date range (should already match, but keep for safety)
    forecast_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(str(start_dt))) & (forecast_df['ds'] <= pd.to_datetime(str(end_dt)))]

    # Format response
    forecast_list = [
        {
            "date": row['ds'].strftime("%Y-%m-%d"),
            "predicted_demand": row['yhat'],
            "lower_bound": row['yhat_lower'],
            "upper_bound": row['yhat_upper']
        }
        for _, row in forecast_df.iterrows()
    ]
    return {
        "status": "success",
        "category": category,
        "start_date": str(start_dt),
        "end_date": str(end_dt),
        "forecast_data": forecast_list
    }


@router.get("/db-forecast", response_model=Dict[str, Any])
def get_db_category_forecast(
    category: str = Query(..., description="Category name"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Fetch demand forecast for a category and date range from the demand_forecasts collection.
    Returns forecast data, current stock, and business-driven recommendations/alerts.
    """
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    collection = get_collection("demand_forecasts")
    # Use yhat, yhat_lower, yhat_upper from user's format
    results = list(collection.find({
        "category": category,
        "date": {"$gte": start_date, "$lte": end_date}
    }, {"_id": 0, "category": 1, "date": 1, "yhat": 1, "yhat_lower": 1, "yhat_upper": 1}).sort("date", 1))
    if not results:
        raise HTTPException(status_code=404, detail="No forecast data found for the given parameters.")

    # Map yhat fields to expected frontend keys
    for d in results:
        d["predicted_demand"] = d.pop("yhat", 0)
        d["lower_bound"] = d.pop("yhat_lower", 0)
        d["upper_bound"] = d.pop("yhat_upper", 0)

    # Fetch current stock from inventory collection (assumes a document per category)
    inventory_collection = get_collection("category_stock")
    stock_doc = inventory_collection.find_one({"category": category}, {"_id": 0, "stock": 1, "stock_level": 1, "total_stock": 1})
    current_stock = 0
    if stock_doc:
        if "stock" in stock_doc:
            current_stock = stock_doc["stock"]
        elif "stock_level" in stock_doc:
            current_stock = stock_doc["stock_level"]
        elif "total_stock" in stock_doc:
            current_stock = stock_doc["total_stock"]

    # Business config thresholds
    thresholds = BUSINESS_CONFIG["inventory_thresholds"]
    safety_stock = current_stock * thresholds["safety_stock_multiplier"]
    reorder_point = current_stock * thresholds["reorder_point_multiplier"]
    max_stock = current_stock * thresholds["max_stock_multiplier"]
    stockout_prob_threshold = thresholds["stockout_probability_threshold"]

    # Calculate total predicted demand for the period
    total_predicted = sum(d.get("predicted_demand", 0) for d in results)
    # Projected stock after forecast period
    projected_stock = current_stock - total_predicted

    # Recommendation and alert logic
    recommendation = None
    alert = None
    if projected_stock < reorder_point:
        recommendation = max(0, reorder_point - projected_stock)
        alert = f"Reorder recommended: Projected stock ({projected_stock:.2f}) below reorder point ({reorder_point:.2f})"
    elif projected_stock < safety_stock:
        alert = f"Warning: Projected stock ({projected_stock:.2f}) below safety stock ({safety_stock:.2f})"
    elif projected_stock > max_stock:
        alert = f"Overstock risk: Projected stock ({projected_stock:.2f}) exceeds max stock ({max_stock:.2f})"

    # Stockout probability (simple: if projected_stock < 0, probability = 1)
    stockout_probability = 1.0 if projected_stock < 0 else 0.0
    if stockout_probability > stockout_prob_threshold:
        alert = f"Stockout risk: Probability ({stockout_probability:.2f}) exceeds threshold ({stockout_prob_threshold})"

    return {
        "category": category,
        "start_date": start_date,
        "end_date": end_date,
        "forecast_data": results,
        "current_stock": current_stock,
        "total_predicted_demand": total_predicted,
        "projected_stock": projected_stock,
        "recommendation": recommendation,
        "alert": alert,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "max_stock": max_stock,
        "stockout_probability": stockout_probability
    }



