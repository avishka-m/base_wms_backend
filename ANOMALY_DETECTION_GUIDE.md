# Advanced Anomaly Detection System

## 🎯 Overview

The Advanced Anomaly Detection System combines rule-based and AI/ML techniques to provide comprehensive anomaly detection for the Warehouse Management System. It uses both threshold-based business rules and Isolation Forest machine learning models to identify unusual patterns and behaviors.

## 🔍 Features

### Dual Detection Approach
- **Rule-based Detection**: Predefined business rules and thresholds
- **ML-based Detection**: Isolation Forest algorithm for statistical anomalies
- **Combined Analysis**: Unified scoring and prioritization

### Detection Categories
1. **📦 Inventory Anomalies**
   - Critical stockouts
   - Extreme low/high stock levels
   - Dead stock identification
   - Statistical inventory patterns

2. **🛒 Order Anomalies**
   - Unusual order timing
   - High-value orders
   - Bulk order patterns
   - Processing delays

3. **🔄 Workflow Anomalies**
   - Stuck orders and processes
   - Workflow bottlenecks
   - Processing delays
   - Stage skipping

4. **👷 Worker Anomalies** (Manager only)
   - Performance anomalies
   - Unusual login patterns
   - Productivity drops
   - Error rate increases

## 🚀 API Endpoints

### Core Detection
- `GET /api/v1/anomaly-detection/health` - Health check
- `GET /api/v1/anomaly-detection/detect` - Comprehensive detection
- `GET /api/v1/anomaly-detection/detect/inventory` - Inventory-specific
- `GET /api/v1/anomaly-detection/detect/orders` - Order-specific
- `GET /api/v1/anomaly-detection/detect/workflow` - Workflow-specific
- `GET /api/v1/anomaly-detection/detect/workers` - Worker-specific (Manager only)

### Analysis & Management
- `GET /api/v1/anomaly-detection/analysis/summary` - Analysis summary
- `POST /api/v1/anomaly-detection/models/retrain` - Retrain ML models (Manager only)
- `GET /api/v1/anomaly-detection/models/status` - Model status (Manager only)
- `GET /api/v1/anomaly-detection/thresholds` - Get thresholds (Manager only)
- `PUT /api/v1/anomaly-detection/thresholds` - Update thresholds (Manager only)

## 🤖 Machine Learning Details

### Isolation Forest
- **Algorithm**: Isolation Forest (scikit-learn)
- **Purpose**: Detect statistical outliers and unusual patterns
- **Features**: Automatically extracted from inventory/order data
- **Training**: Retrainable with latest data
- **Contamination Rate**: Configurable (default: 10%)

### Model Management
- **Persistence**: Models saved using joblib
- **Retraining**: Background process, estimated 5-10 minutes
- **Status Monitoring**: Performance metrics and training dates
- **Feature Engineering**: Automatic feature extraction from data

## 📊 Detection Process

### Rule-based Detection
1. **Threshold Checks**: Compare values against configurable thresholds
2. **Business Logic**: Apply warehouse-specific business rules
3. **Pattern Recognition**: Identify known problematic patterns
4. **Severity Scoring**: Assign severity levels (critical, high, medium, low)

### ML-based Detection
1. **Data Collection**: Gather recent inventory/order data
2. **Feature Engineering**: Extract relevant numerical features
3. **Anomaly Scoring**: Use trained Isolation Forest models
4. **Threshold Application**: Convert scores to anomaly classifications
5. **Result Formatting**: Structure results for consistency

### Combined Analysis
1. **Result Merging**: Combine rule-based and ML results
2. **Deduplication**: Remove overlapping detections
3. **Priority Scoring**: Unified severity and priority scoring
4. **Recommendation Generation**: Actionable recommendations
5. **Summary Statistics**: Overall health and trend analysis

## ⚙️ Configuration

### Rule Thresholds
```python
rule_thresholds = {
    "inventory": {
        "critical_stock_threshold": 5,
        "low_stock_threshold": 20,
        "high_stock_multiplier": 10,
        "dead_stock_days": 90
    },
    "orders": {
        "high_value_threshold": 10000,
        "bulk_quantity_threshold": 1000,
        "processing_delay_hours": 24
    },
    "workflow": {
        "stuck_order_hours": 48,
        "bottleneck_threshold": 10
    },
    "workers": {
        "low_performance_threshold": 0.7,
        "error_rate_threshold": 0.1
    }
}
```

### ML Parameters
```python
ml_params = {
    "contamination": 0.1,  # Expected anomaly rate
    "random_state": 42,    # Reproducibility
    "n_estimators": 100    # Number of trees
}
```

## 🔒 Security & Permissions

### Role-based Access
- **All Users**: Basic detection and health checks
- **Managers Only**: 
  - Worker anomaly detection
  - Model retraining
  - Threshold management
  - Model status monitoring

### Authentication
- JWT token required for all endpoints
- Role validation for restricted endpoints
- Audit logging for management operations

## 📈 Usage Examples

### Basic Detection
```python
# Get comprehensive anomalies
GET /api/v1/anomaly-detection/detect?include_ml=true

# Get inventory anomalies only (rule-based)
GET /api/v1/anomaly-detection/detect/inventory?technique=rule

# Get inventory anomalies (both techniques)
GET /api/v1/anomaly-detection/detect/inventory?technique=both
```

### Management Operations
```python
# Retrain ML models (Manager only)
POST /api/v1/anomaly-detection/models/retrain

# Update thresholds (Manager only)
PUT /api/v1/anomaly-detection/thresholds
{
  "inventory": {
    "critical_stock_threshold": 3
  }
}

# Get analysis summary
GET /api/v1/anomaly-detection/analysis/summary?days=30
```

## 🛠️ Installation & Setup

### Dependencies
```bash
# Core ML libraries (already included in requirements.txt)
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.5.1
```

### Testing
```bash
# Run the test script
python test_anomaly_detection.py
```

### Verification
1. Start the FastAPI server
2. Check health endpoint: `GET /api/v1/anomaly-detection/health`
3. Run comprehensive detection: `GET /api/v1/anomaly-detection/detect`
4. Verify all detection categories work correctly

## 🔍 Troubleshooting

### Common Issues

1. **ML Model Training Fails**
   - Ensure sufficient data exists in inventory/orders collections
   - Check database connectivity
   - Verify feature extraction works correctly

2. **High False Positive Rate**
   - Adjust contamination rate in ML parameters
   - Fine-tune rule-based thresholds
   - Analyze data quality and patterns

3. **Performance Issues**
   - Consider data sampling for large datasets
   - Optimize database queries
   - Use background processing for heavy operations

### Error Handling
- Graceful degradation: Rule-based detection continues if ML fails
- Detailed error logging for debugging
- User-friendly error messages in API responses

## 📚 Technical Architecture

### Service Layer
- `AdvancedAnomalyDetectionService`: Main service class
- `rule_thresholds`: Configurable threshold dictionary
- `ml_models`: Isolation Forest model storage

### API Layer
- FastAPI routers with comprehensive endpoints
- Role-based access control
- Background task support for model retraining

### Data Layer
- MongoDB integration for real-time data access
- Feature engineering pipelines
- Model persistence using joblib

## 🎯 Future Enhancements

1. **Advanced ML Models**
   - Time series anomaly detection
   - Deep learning approaches
   - Ensemble methods

2. **Real-time Detection**
   - Stream processing integration
   - WebSocket notifications
   - Dashboard integration

3. **Predictive Analytics**
   - Anomaly forecasting
   - Trend analysis
   - Preventive recommendations

4. **Enhanced Visualization**
   - Anomaly heatmaps
   - Interactive dashboards
   - Historical trend charts

## 📞 Support

For questions or issues with the anomaly detection system:
1. Check this documentation
2. Run the test script for diagnostic information
3. Review API endpoint responses for detailed error messages
4. Contact the development team for advanced troubleshooting
