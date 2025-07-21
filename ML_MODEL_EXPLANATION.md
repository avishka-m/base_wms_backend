# 🤖 Machine Learning Models in Anomaly Detection System
## Complete Training & Implementation Guide

## 📚 Overview: What We Built

Our system uses **Isolation Forest** - an unsupervised machine learning algorithm that's perfect for anomaly detection. Here's exactly how it works in our warehouse system:

## 🎯 1. What is Isolation Forest?

### The Concept (Simple Explanation)
```
Imagine you have a forest of decision trees.
Normal data points are hard to "isolate" (like finding a needle in a haystack).
Anomalous data points are easy to "isolate" (they stick out).

Example: 
- Normal inventory: 50-200 units
- Anomaly: 50,000 units (easy to isolate!)
```

### Technical Details
- **Unsupervised**: No need for labeled "anomaly" vs "normal" examples
- **Tree-based**: Uses random decision trees to isolate data points
- **Scoring**: Gives each data point an anomaly score (-1 to 1)
- **Threshold**: Points with score < 0 are considered anomalies

## 🔧 2. How Our Models Are Trained

### Step 1: Data Collection
```python
# From our code in detect_inventory_ml_anomalies()
inventory_collection = self.db["inventory"]
items = await inventory_collection.find({}).to_list(None)

# We need at least 10 items for training
if len(items) < 10:
    logger.warning("Insufficient inventory data for ML anomaly detection")
    return []
```

### Step 2: Feature Engineering
```python
# We extract 7 features from each inventory item
for item in items:
    current_stock = item.get("stock_quantity", 0)
    min_stock = item.get("min_stock_level", 1)
    max_stock = item.get("max_stock_level", 100)
    price = item.get("price", 0)
    
    # Calculate derived features
    stock_ratio = current_stock / max(min_stock, 1)
    stock_range_position = (current_stock - min_stock) / max(max_stock - min_stock, 1)
    value_at_risk = current_stock * price
    
    feature_vector = [
        current_stock,          # Raw stock level
        stock_ratio,            # Stock vs minimum ratio
        stock_range_position,   # Position in min-max range
        value_at_risk,          # Total value of stock
        price,                  # Item price
        min_stock,              # Minimum stock level
        max_stock               # Maximum stock level
    ]
```

### Step 3: Data Preprocessing
```python
# Convert to numpy array and scale features
X = np.array(features)  # Convert list to numpy array
X_scaled = self.scalers["inventory"].fit_transform(X)  # Scale features to similar ranges
```

**Why scaling?** Different features have different scales:
- Stock quantity: 0-1000
- Price: $1-$500
- Ratios: 0.0-1.0

Scaling makes them comparable.

### Step 4: Model Training
```python
# Train the Isolation Forest model
self.models["inventory"].fit(X_scaled)

# The model learns what "normal" inventory patterns look like
# It builds 100 random trees (n_estimators=100)
# Each tree randomly isolates data points
```

### Step 5: Anomaly Detection
```python
# Get anomaly scores and predictions
anomaly_scores = self.models["inventory"].decision_function(X_scaled)
predictions = self.models["inventory"].predict(X_scaled)

# predictions: +1 = normal, -1 = anomaly
# anomaly_scores: lower scores = more anomalous
```

## 📊 3. Model Parameters Explained

```python
# From our advanced_anomaly_detection_service.py
self.ml_params = {
    "contamination": 0.1,      # Expect 10% of data to be anomalies
    "random_state": 42,        # For reproducible results
    "n_estimators": 100,       # Number of decision trees
    "max_samples": "auto"      # Automatic sample size
}
```

### What Each Parameter Does:

1. **contamination=0.1**: 
   - Tells the model to expect 10% anomalies
   - Adjusts the threshold for anomaly detection
   - Lower = fewer anomalies detected, Higher = more anomalies

2. **random_state=42**: 
   - Makes results reproducible
   - Same data = same results every time

3. **n_estimators=100**: 
   - Uses 100 decision trees
   - More trees = better accuracy, slower training

4. **max_samples="auto"**: 
   - Automatically determines training sample size
   - Good for performance with large datasets

## 🏗️ 4. Complete Training Process (Step by Step)

### When Training Happens:
```python
# Training happens EVERY TIME you call detect_inventory_ml_anomalies()
async def detect_inventory_ml_anomalies(self):
    # 1. Get fresh data from database
    items = await inventory_collection.find({}).to_list(None)
    
    # 2. Extract features
    features = [extract_features(item) for item in items]
    
    # 3. Scale features
    X_scaled = self.scalers["inventory"].fit_transform(features)
    
    # 4. Train model on current data
    self.models["inventory"].fit(X_scaled)
    
    # 5. Immediately predict anomalies
    predictions = self.models["inventory"].predict(X_scaled)
```

### Alternative: Persistent Training
```python
# For production, you can save trained models
async def save_models(self):
    for model_type, model in self.models.items():
        model_path = f"{self.model_dir}/{model_type}_model.joblib"
        scaler_path = f"{self.model_dir}/{model_type}_scaler.joblib"
        
        joblib.dump(model, model_path)           # Save trained model
        joblib.dump(self.scalers[model_type], scaler_path)  # Save scaler

async def load_models(self):
    # Load previously trained models
    for model_type in self.models.keys():
        model_path = f"{self.model_dir}/{model_type}_model.joblib"
        if os.path.exists(model_path):
            self.models[model_type] = joblib.load(model_path)
```

## 🎯 5. Real Example: How It Detects Anomalies

### Sample Inventory Data:
```python
items = [
    {"stock_quantity": 100, "min_stock": 50, "max_stock": 200, "price": 10},   # Normal
    {"stock_quantity": 150, "min_stock": 50, "max_stock": 200, "price": 15},   # Normal  
    {"stock_quantity": 75,  "min_stock": 50, "max_stock": 200, "price": 12},   # Normal
    {"stock_quantity": 50000, "min_stock": 50, "max_stock": 200, "price": 10}, # ANOMALY!
    {"stock_quantity": 120, "min_stock": 50, "max_stock": 200, "price": 8},    # Normal
]
```

### Feature Extraction:
```python
# Item 1 (Normal): [100, 2.0, 0.33, 1000, 10, 50, 200]
# Item 4 (Anomaly): [50000, 1000, 331.0, 500000, 10, 50, 200]  # Extreme values!
```

### Model Results:
```python
predictions = [1, 1, 1, -1, 1]      # -1 indicates anomaly
scores = [0.1, 0.15, 0.05, -0.8, 0.12]  # -0.8 is very anomalous
```

## 🔄 6. How to Retrain Models

### Automatic Retraining (Current Implementation):
- Models retrain every time detection runs
- Always uses latest data
- Good for dynamic environments

### Manual Retraining (via API):
```bash
# Call the retrain endpoint
POST /api/v1/anomaly-detection/models/retrain

# This runs in background and saves updated models
```

### Scheduled Retraining:
```python
# You could add this to run daily
async def scheduled_retrain():
    # Collect last 30 days of data
    # Train models
    # Save updated models
    # Update performance metrics
```

## 📈 7. Model Performance & Evaluation

### How We Measure Success:
```python
# In production, you'd track:
performance_metrics = {
    "anomalies_detected": 15,
    "false_positives": 2,      # Normal items flagged as anomalies
    "false_negatives": 1,      # Anomalies missed
    "precision": 0.87,         # 87% of detected anomalies are real
    "recall": 0.93,           # 93% of real anomalies detected
    "contamination_rate": 0.1  # Expected vs actual anomaly rate
}
```

### Improving Performance:
1. **Adjust contamination rate** based on actual anomaly frequency
2. **Add more features** (seasonality, supplier info, etc.)
3. **Use historical data** for better training
4. **Ensemble methods** (combine multiple algorithms)

## 🛠️ 8. Current Implementation Status

### What's Working:
✅ **Feature Engineering**: 7 meaningful features extracted  
✅ **Model Training**: Isolation Forest properly configured  
✅ **Scaling**: StandardScaler normalizes features  
✅ **Anomaly Detection**: Returns scores and classifications  
✅ **Model Persistence**: Save/load functionality implemented  

### What Needs Real Data:
⚠️ **Historical Patterns**: Currently trains on current snapshot  
⚠️ **Seasonal Trends**: No time-series features yet  
⚠️ **Performance Metrics**: Need real anomalies to validate  

## 🚀 9. How to Use in Production

### Step 1: Accumulate Training Data
```python
# Let system run for 2-4 weeks to collect normal patterns
# This gives the model a good baseline of "normal" behavior
```

### Step 2: Initial Training
```python
# Train models on historical data
await advanced_anomaly_detection_service.detect_all_anomalies(include_ml=True)
```

### Step 3: Save Models
```python
# Save trained models for reuse
await advanced_anomaly_detection_service.save_models()
```

### Step 4: Monitor & Retrain
```python
# Weekly or monthly retraining
# Monitor false positive/negative rates
# Adjust contamination parameter as needed
```

## 🎯 10. Why This Approach Works

### Advantages:
1. **No labeled data needed** - learns from patterns automatically
2. **Adapts to your data** - learns what's normal for YOUR warehouse
3. **Handles multiple features** - considers relationships between variables
4. **Fast detection** - real-time anomaly scoring
5. **Configurable sensitivity** - adjust contamination rate

### Use Cases Perfect For:
- Unusual stock quantities
- Abnormal order patterns  
- Equipment sensor anomalies
- Performance outliers
- Data quality issues

## 📝 Summary: The Complete ML Pipeline

```
1. 📥 Data Collection    → Get inventory/order data from MongoDB
2. 🔧 Feature Engineering → Extract 7 meaningful features per item
3. 📊 Data Preprocessing → Scale features to comparable ranges
4. 🤖 Model Training     → Isolation Forest learns normal patterns
5. 🎯 Anomaly Detection  → Score each item, flag anomalies
6. 📋 Result Formatting  → Return structured anomaly reports
7. 💾 Model Persistence  → Save models for future use
8. 🔄 Continuous Learning → Retrain with new data periodically
```

The beauty of this system is that it **learns automatically** from your warehouse data and gets better over time. No need to manually define what's "normal" - the ML models figure it out from the patterns in your data!

Want me to explain any specific part in more detail or show you how to customize it for your specific needs?
