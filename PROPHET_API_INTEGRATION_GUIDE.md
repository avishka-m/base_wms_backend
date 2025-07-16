# Prophet Forecasting API Integration

This document explains how to use the Prophet forecasting API endpoints that connect the backend Prophet models to frontend user interfaces.

## Overview

The Prophet forecasting API provides a user-friendly interface for:
- Getting demand forecasts for specific products
- Customizing forecast time ranges
- Training models for products
- Getting inventory recommendations
- Managing model status

## API Endpoints

### Base URL
```
http://localhost:8000/api/prophet
```

### Authentication
All endpoints require appropriate user roles:
- **Warehouse Staff, Analyst, Manager**: Can view forecasts and products
- **Analyst, Manager**: Can generate forecasts and recommendations
- **Manager**: Can train models

---

## 1. Health Check

Check if the Prophet forecasting service is running.

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "prophet-forecasting",
  "service_status": {
    "status": "available",
    "models_available": 15,
    "models_path": "/path/to/models"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 2. Get Available Products

Get list of products available for forecasting.

**GET** `/products`

**Response:**
```json
{
  "status": "success",
  "total_products": 25,
  "products_with_models": 15,
  "products": [
    {
      "product_id": "PROD_001",
      "has_trained_model": true,
      "data_points": 365,
      "last_data_date": "2024-01-14"
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 3. Single Product Forecast

Generate a forecast for a specific product.

**POST** `/forecast/single`

**Request Body:**
```json
{
  "product_id": "PROD_001",
  "horizon_days": 30,
  "start_date": "2024-01-16",  // Optional, defaults to tomorrow
  "confidence_interval": 0.95,
  "include_external_factors": true
}
```

**Response:**
```json
{
  "status": "success",
  "item_id": "PROD_001",
  "forecast_period": {
    "start_date": "2024-01-16",
    "end_date": "2024-02-14",
    "days": 30
  },
  "predictions": [
    {
      "date": "2024-01-16",
      "predicted_demand": 45.23,
      "lower_bound": 38.15,
      "upper_bound": 52.31,
      "confidence": 0.95
    }
  ],
  "summary": {
    "total_predicted_demand": 1356.90,
    "average_daily_demand": 45.23,
    "peak_daily_demand": 67.45,
    "confidence_interval": 0.95
  },
  "model_info": {
    "model_type": "Prophet",
    "trained_on": "2024-01-15 10:30:00",
    "model_file": "/path/to/PROD_001_prophet_model.pkl"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 4. Custom Date Range Forecast

Generate a forecast for a specific date range.

**POST** `/forecast/custom-range`

**Request Body:**
```json
{
  "product_id": "PROD_001",
  "start_date": "2024-02-01",
  "end_date": "2024-02-14",
  "confidence_interval": 0.90
}
```

**Response:** Similar to single product forecast with additional `custom_range` field.

---

## 5. Batch Forecast

Generate forecasts for multiple products.

**POST** `/forecast/batch`

**Request Body:**
```json
{
  "product_ids": ["PROD_001", "PROD_002", "PROD_003"],
  "horizon_days": 30,
  "start_date": "2024-01-16",  // Optional
  "confidence_interval": 0.95,
  "include_external_factors": true
}
```

**Response:**
```json
{
  "status": "success",
  "batch_summary": {
    "total_items": 3,
    "successful": 3,
    "failed": 0,
    "success_rate": "100.0%"
  },
  "results": {
    "PROD_001": { /* forecast result */ },
    "PROD_002": { /* forecast result */ },
    "PROD_003": { /* forecast result */ }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

For large batches (>20 products), processing happens in background:
```json
{
  "status": "processing",
  "task_id": "prophet_batch_20240115_103000",
  "product_count": 50,
  "estimated_completion": "2024-01-15T11:20:00Z",
  "message": "Large batch processing in background. Use task_id to check status."
}
```

---

## 6. Train Models

Train Prophet models for specific products.

**POST** `/models/train`

**Request Body:**
```json
{
  "product_ids": ["PROD_001"],  // Optional, empty = all products
  "retrain_existing": false
}
```

**Response:**
```json
{
  "status": "training_started",
  "task_id": "prophet_train_20240115_103000",
  "product_count": 1,
  "retrain_existing": false,
  "estimated_completion": "2024-01-15T11:00:00Z",
  "message": "Prophet model training started in background"
}
```

---

## 7. Inventory Recommendations

Get AI-powered inventory recommendations.

**POST** `/recommendations/inventory`

**Request Body:**
```json
{
  "forecast_days": 30,
  "min_confidence": 0.8,
  "product_filter": null,  // Optional: specific products
  "include_seasonal_analysis": true
}
```

**Response:**
```json
{
  "status": "success",
  "recommendations": [
    {
      "product_id": "PROD_001",
      "action": "increase_stock",
      "priority": "high",
      "predicted_total_demand": 1356.90,
      "predicted_avg_daily_demand": 45.23,
      "confidence": 0.8,
      "recommendation_period": "30 days"
    }
  ],
  "analysis_period_days": 30,
  "min_confidence": 0.8,
  "total_products_analyzed": 10,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 8. Model Status

Get status of trained models.

**GET** `/models/status`

**Response:**
```json
{
  "service_status": {
    "status": "available",
    "models_available": 15,
    "models_path": "/path/to/models"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## Frontend Integration Examples

### JavaScript/React Example

```javascript
// Single product forecast
const forecast = async (productId, days = 30) => {
  try {
    const response = await fetch('/api/prophet/forecast/single', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token
      },
      body: JSON.stringify({
        product_id: productId,
        horizon_days: days,
        confidence_interval: 0.95
      })
    });
    
    const data = await response.json();
    
    if (data.status === 'success') {
      console.log('Total demand:', data.summary.total_predicted_demand);
      return data.predictions;
    } else {
      throw new Error(data.message);
    }
  } catch (error) {
    console.error('Forecast error:', error);
  }
};

// Usage
forecast('PROD_001', 30).then(predictions => {
  predictions.forEach(pred => {
    console.log(`${pred.date}: ${pred.predicted_demand} units`);
  });
});
```

### Python Example

```python
import requests

API_BASE = "http://localhost:8000/api/prophet"

def get_forecast(product_id, days=30):
    """Get forecast for a product"""
    url = f"{API_BASE}/forecast/single"
    payload = {
        "product_id": product_id,
        "horizon_days": days,
        "confidence_interval": 0.95
    }
    
    response = requests.post(url, json=payload)
    data = response.json()
    
    if data["status"] == "success":
        return data["predictions"]
    else:
        raise Exception(f"Forecast failed: {data['message']}")

# Usage
predictions = get_forecast("PROD_001", 30)
for pred in predictions:
    print(f"{pred['date']}: {pred['predicted_demand']:.2f} units")
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error message description",
  "status_code": 400  // or 404, 500, 503
}
```

Common error codes:
- **400**: Bad request (invalid parameters)
- **404**: Product not found or no data
- **500**: Internal server error
- **503**: Service unavailable

---

## Performance Notes

1. **First-time forecasts** may take longer as models are trained on demand
2. **Cached models** provide faster responses for subsequent requests
3. **Large batch requests** (>20 products) are processed in background
4. **Model training** happens in background for multiple products

---

## Testing

Use the provided test files:

1. **API Tests**: `test_prophet_api.py`
   ```bash
   python test_prophet_api.py
   ```

2. **Frontend Demo**: `prophet_dashboard.html`
   - Open in browser
   - Connects to local API server
   - Interactive forecast generation

---

## Configuration

Key configuration files:
- `ai_services/seasonal_inventory/config.py` - Prophet and data settings
- `app/services/prophet_forecasting_service.py` - Service configuration
- `app/api/prophet_forecasting.py` - API endpoint configuration

---

## Next Steps

1. **Integrate with your frontend framework** (React, Vue, Angular)
2. **Add authentication headers** based on your auth system
3. **Customize forecast parameters** for your business needs
4. **Add data visualization** using Chart.js, D3.js, or similar
5. **Implement caching** for frequently requested forecasts
6. **Add scheduling** for automated model retraining
