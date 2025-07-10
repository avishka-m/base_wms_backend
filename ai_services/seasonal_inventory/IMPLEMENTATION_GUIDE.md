# Seasonal Inventory Prediction - Implementation Guide

## Step-by-Step Implementation

### Step 1: Environment Setup

#### 1.1 Install Dependencies
```bash
cd backend/ai-services/seasonal-inventory
pip install -r requirements.txt
```

#### 1.2 Configure API Keys (Run in your activated virtual environment terminal)
```bash
# Setup Kaggle API (run this in your terminal where venv is activated)
kaggle config set username YOUR_KAGGLE_USERNAME
kaggle config set key YOUR_KAGGLE_KEY

# For Windows PowerShell - Setup environment variables
$env:OPENAI_API_KEY="your_openai_key"
$env:MONGODB_URL="mongodb://localhost:27017"
$env:WMS_API_BASE_URL="http://localhost:8000/api/v1"

# Or create a .env file in the seasonal-inventory directory
# .env file content:
# OPENAI_API_KEY=your_openai_key
# MONGODB_URL=mongodb://localhost:27017
# WMS_API_BASE_URL=http://localhost:8000/api/v1
```

### Step 2: Data Collection Strategy

#### 2.1 Kaggle Datasets Priority List
```python
DATASETS = {
    # High Priority - Retail & Inventory
    "carrie1/ecommerce-data": {
        "description": "E-commerce transaction data with seasonal patterns",
        "size": "~45MB",
        "features": ["InvoiceDate", "Quantity", "UnitPrice", "StockCode"],
        "seasonality": "Strong yearly/monthly patterns"
    },
    
    "mkechinov/ecommerce-behavior-data": {
        "description": "E-commerce behavior with time-based patterns",
        "size": "~2GB", 
        "features": ["event_time", "product_id", "category_code", "brand"],
        "seasonality": "Daily/weekly patterns"
    },
    
    "olistbr/brazilian-ecommerce": {
        "description": "Brazilian e-commerce public dataset",
        "size": "~200MB",
        "features": ["order_purchase_timestamp", "product_category"],
        "seasonality": "Regional seasonal patterns"
    },
    
    # Medium Priority - Supply Chain
    "shashwatwork/dataco-smart-supply-chain": {
        "description": "Supply chain dataset with demand patterns",
        "size": "~180MB",
        "features": ["order_date", "delivery_date", "product_name"],
        "seasonality": "Supply chain seasonality"
    },
    
    "prasad22/retail-transactions-dataset": {
        "description": "Retail transactions with seasonal trends",
        "size": "~50MB",
        "features": ["Transaction_Date", "Product_Category", "Quantity"],
        "seasonality": "Strong seasonal components"
    }
}
```

#### 2.2 Data Collection Workflow
```
1. Kaggle Data Download → 2. WMS Historical Data → 3. External APIs → 4. Data Validation
     ↓                          ↓                       ↓                ↓
   Raw CSV Files        MongoDB Export          Weather/Holiday Data   Quality Checks
     ↓                          ↓                       ↓                ↓
   5. Data Standardization → 6. Feature Engineering → 7. Prophet Format → 8. Model Training
```

### Step 3: Data Processing Pipeline

#### 3.1 Data Schema Standardization
```python
STANDARD_SCHEMA = {
    "ds": "datetime64[ns]",  # Prophet required date column
    "y": "float64",          # Prophet required target variable
    "product_id": "str",     # Product identifier
    "category": "str",       # Product category
    "quantity": "int64",     # Quantity sold/demanded
    "price": "float64",      # Unit price
    "total_value": "float64", # Total transaction value
    "warehouse_id": "str",   # Warehouse location
    "season": "str",         # Season indicator
    "holiday": "bool",       # Holiday flag
    "weather": "float64",    # Weather indicator
    "promotion": "bool"      # Promotion flag
}
```

#### 3.2 Feature Engineering Strategy
```python
FEATURES = {
    "temporal": [
        "year", "month", "day", "dayofweek", "quarter",
        "is_weekend", "is_month_end", "is_quarter_end"
    ],
    "seasonal": [
        "season_spring", "season_summer", "season_autumn", "season_winter"
    ],
    "external": [
        "holiday_effect", "weather_temp", "weather_rain", "economic_index"
    ],
    "business": [
        "promotion_active", "new_product_launch", "inventory_level"
    ],
    "lag_features": [
        "demand_lag_7", "demand_lag_30", "demand_lag_365"
    ]
}
```

### Step 4: Prophet Model Configuration

#### 4.1 Model Architecture
```python
MODEL_CONFIG = {
    "base_model": {
        "growth": "linear",
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False
    },
    
    "custom_seasonalities": [
        {"name": "monthly", "period": 30.5, "fourier_order": 5},
        {"name": "quarterly", "period": 91.25, "fourier_order": 3},
        {"name": "bi_yearly", "period": 182.5, "fourier_order": 2}
    ],
    
    "holidays": {
        "include_country_holidays": ["US", "BR", "GB"],
        "custom_holidays": [
            "Black Friday", "Cyber Monday", "Back to School",
            "Mother's Day", "Father's Day", "Valentine's Day"
        ]
    },
    
    "hyperparameters": {
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5],
        "seasonality_prior_scale": [0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.1, 1.0, 10.0]
    }
}
```

#### 4.2 Model Training Strategy
```python
TRAINING_STRATEGY = {
    "cross_validation": {
        "initial": "730 days",  # 2 years initial training
        "period": "90 days",    # Retrain every 3 months
        "horizon": "90 days"    # Forecast 3 months ahead
    },
    
    "validation_split": {
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "test_ratio": 0.1
    },
    
    "ensemble": {
        "models": ["prophet_base", "prophet_tuned", "seasonal_naive"],
        "weights": [0.6, 0.3, 0.1],
        "combination_method": "weighted_average"
    }
}
```

### Step 5: Visualization Dashboard

#### 5.1 Dashboard Components
```python
DASHBOARD_COMPONENTS = {
    "forecast_charts": {
        "trend_decomposition": "Time series decomposition plot",
        "seasonal_patterns": "Seasonal component visualization",
        "forecast_uncertainty": "Prediction intervals",
        "model_performance": "Accuracy metrics over time"
    },
    
    "business_kpis": {
        "inventory_turnover": "Current vs predicted turnover",
        "stockout_risk": "Probability of stockout by product",
        "reorder_recommendations": "Automated reorder suggestions",
        "cost_savings": "Projected cost savings from optimization"
    },
    
    "interactive_filters": {
        "product_category": "Filter by product category",
        "warehouse_location": "Filter by warehouse",
        "time_range": "Adjustable forecast horizon",
        "confidence_level": "Prediction interval adjustment"
    }
}
```

#### 5.2 Visualization Examples

**Trend Decomposition Chart:**
```
Demand Time Series Decomposition
┌─────────────────────────────────────────────────────────────┐
│ Original Data        📈 Showing actual demand over time     │
├─────────────────────────────────────────────────────────────┤
│ Trend Component     📊 Long-term trend (growing/declining)  │
├─────────────────────────────────────────────────────────────┤
│ Seasonal Component  🔄 Repeating seasonal patterns          │
├─────────────────────────────────────────────────────────────┤
│ Holiday Component   🎄 Holiday effects and special events   │
├─────────────────────────────────────────────────────────────┤
│ Residual Component  📉 Unexplained variation               │
└─────────────────────────────────────────────────────────────┘
```

**Seasonal Pattern Visualization:**
```
Yearly Seasonality Pattern
   Demand Multiplier
        1.4 |     *
            |    * *
        1.2 |   *   *
            |  *     *
        1.0 |.*       *.
            |           *
        0.8 |            *
            |             *
        0.6 |              *
            └─────────────────────────
            Jan  Apr  Jul  Oct  Dec
            
Weekly Seasonality Pattern
   Demand Multiplier  
        1.3 |    
            |  *   *
        1.1 | * * * *
            |*       *
        0.9 |         *
            |          *
        0.7 |           *
            └─────────────────
            Mon Tue Wed Thu Fri Sat Sun
```

### Step 6: API Integration

#### 6.1 REST API Endpoints
```python
API_ENDPOINTS = {
    "forecasting": {
        "/api/forecast/product/{product_id}": "Get forecast for specific product",
        "/api/forecast/category/{category}": "Get forecast for product category",
        "/api/forecast/warehouse/{warehouse_id}": "Get warehouse-level forecast",
        "/api/forecast/batch": "Batch forecast for multiple products"
    },
    
    "analytics": {
        "/api/analytics/seasonality": "Get seasonal patterns analysis",
        "/api/analytics/trends": "Get trend analysis",
        "/api/analytics/accuracy": "Get model accuracy metrics",
        "/api/analytics/alerts": "Get inventory alerts and recommendations"
    },
    
    "model_management": {
        "/api/model/retrain": "Trigger model retraining",
        "/api/model/status": "Get model training status",
        "/api/model/config": "Get/update model configuration",
        "/api/model/health": "Model health check"
    }
}
```

#### 6.2 Sample API Response
```json
{
    "product_id": "SKU001",
    "forecast_date": "2025-06-25",
    "forecast_horizon": 90,
    "predictions": [
        {
            "date": "2025-06-26",
            "predicted_demand": 156.7,
            "lower_bound": 98.2,
            "upper_bound": 215.2,
            "confidence_interval": 0.8,
            "seasonal_component": 1.12,
            "trend_component": 140.2,
            "holiday_effect": 0.0
        }
    ],
    "seasonality_insights": {
        "peak_season": "November-December",
        "low_season": "January-February",
        "weekly_pattern": "Higher demand on weekends",
        "monthly_pattern": "End-of-month surge"
    },
    "recommendations": {
        "reorder_point": 450,
        "safety_stock": 120,
        "next_reorder_date": "2025-07-15",
        "suggested_quantity": 800
    },
    "model_metadata": {
        "model_version": "v1.2.3",
        "training_date": "2025-06-20",
        "accuracy_score": 0.92,
        "data_freshness": "2 hours ago"
    }
}
```

### Step 6.3: API Key Management for React + REST API

To securely use API keys (e.g., for OpenAI, Kaggle, or internal services) in your React + REST API setup:

1. **Backend (FastAPI) Configuration:**
   - Store all sensitive API keys in environment variables or a `.env` file (never hardcode in code or expose to frontend).
   - Example `.env`:
     ```env
     OPENAI_API_KEY=your_openai_key
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_key
     MONGODB_URL=mongodb://localhost:27017
     WMS_API_BASE_URL=http://localhost:8000/api/v1
     ```
   - Load these in FastAPI using `os.environ` or a library like `python-dotenv`.

2. **Backend Usage:**
   - Use the keys only in backend service calls (e.g., when calling OpenAI or Kaggle APIs from FastAPI endpoints).
   - Never send API keys to the frontend or include them in API responses.

3. **Frontend (React) Usage:**
   - The React app should only call your backend REST API endpoints (e.g., `/api/forecast/...`).
   - Never include or expose API keys in React code or in the browser.

4. **Example FastAPI Key Loading:**
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
   KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
   KAGGLE_KEY = os.getenv('KAGGLE_KEY')
   # Use these in your backend service logic only
   ```

5. **Security Note:**
   - If you deploy, make sure your deployment environment (e.g., Docker, cloud) also sets these environment variables securely.

---

This ensures your API keys are never exposed to the frontend and are only used server-side for secure API calls.

### Step 7: Implementation Checklist

#### Phase 1: Foundation (Week 1-2)
- [ ] Setup project structure
- [ ] Install dependencies
- [ ] Configure API keys (Kaggle, OpenAI)
- [ ] Download initial datasets
- [ ] Setup MongoDB connection
- [ ] Create data validation pipeline
- [ ] Implement basic data preprocessing

#### Phase 2: Model Development (Week 3-4)
- [ ] Implement Prophet model wrapper
- [ ] Create seasonal analysis tools
- [ ] Setup cross-validation framework
- [ ] Implement hyperparameter tuning
- [ ] Create model evaluation metrics
- [ ] Setup automated retraining pipeline

#### Phase 3: Visualization (Week 5)
- [ ] Create forecast visualization charts
- [ ] Implement seasonal pattern plots
- [ ] Build interactive dashboard
- [ ] Setup real-time data updates
- [ ] Create mobile-responsive design

#### Phase 4: Integration (Week 6)
- [ ] Develop REST API endpoints
- [ ] Integrate with existing WMS database
- [ ] Setup authentication and permissions
- [ ] Create frontend integration hooks
- [ ] Implement caching layer

#### Phase 5: Testing & Deployment (Week 7-8)
- [ ] Unit tests for all components
- [ ] Integration tests with WMS
- [ ] Performance testing
- [ ] Setup monitoring and logging
- [ ] Deploy to production environment
- [ ] Create user documentation

### Step 8: Next Actions

1. **Start with Data Collection**
   ```bash
   # Navigate to seasonal inventory directory
   cd backend/ai-services/seasonal-inventory
   
   # Create directory structure
   mkdir -p {data/{datasets,processed,models},src/{data_collection,preprocessing,models,visualization,api},notebooks,tests}
   
   # Install requirements
   pip install -r requirements.txt
   ```

2. **Setup Kaggle Integration**
   ```python
   # First script to implement
   from kaggle.api.kaggle_api_extended import KaggleApi
   
   api = KaggleApi()
   api.authenticate()
   
   # Download priority datasets
   api.dataset_download_files('carrie1/ecommerce-data', path='data/datasets/', unzip=True)
   ```

3. **Create Initial Prophet Model**
   ```python
   # Basic Prophet implementation
   from prophet import Prophet
   import pandas as pd
   
   # Load and prepare data
   df = pd.read_csv('data/datasets/ecommerce_data.csv')
   df_prophet = df[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
   
   # Train model
   model = Prophet()
   model.fit(df_prophet)
   
   # Generate forecast
   future = model.make_future_dataframe(periods=90)
   forecast = model.predict(future)
   ```

Would you like me to proceed with implementing any specific component first, or would you prefer to start with the data collection and Kaggle integration?
