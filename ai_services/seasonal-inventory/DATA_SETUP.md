# 📊 Data Setup for Seasonal Inventory Prediction

## 🚨 Important: Data Files Not Included in Git

Due to GitHub file size limitations, the training data files are **NOT included** in the repository. You need to set up the data locally.

## 📁 Required Data Structure

The AI service expects the following data structure:

```
ai-services/seasonal-inventory/
├── data/
│   ├── processed/
│   │   ├── daily_demand_by_product.csv          # Legacy data (2010-2011)
│   │   └── daily_demand_by_product_modern.csv   # Modern data (2022-2024) ⭐
│   ├── datasets/
│   │   └── [raw dataset files]
│   └── models/
│       └── [trained Prophet models - auto-generated]
```

## 🎯 Data Generation

### Option 1: Generate Modern Data (Recommended)
```bash
cd ai-services/seasonal-inventory
python create_modern_data.py
```

This will generate:
- **2,192,000 records** of modern e-commerce data
- **Date range**: 2022-01-01 to 2024-12-31
- **6 product categories** with realistic patterns
- **File**: `data/processed/daily_demand_by_product_modern.csv`

### Option 2: Use Legacy Data
If you have the original dataset, place it at:
- `data/processed/daily_demand_by_product.csv`

## 🔄 How the Service Works

1. **Auto-Detection**: Service automatically uses modern data if available
2. **Fallback**: Falls back to legacy data if modern data not found
3. **Training**: Prophet models train in real-time (no pre-training needed)

## ✅ Verification

Check if data is properly set up:
```bash
# Test the health endpoint
curl http://localhost:8002/api/v1/predictions/health
```

Expected response should show:
```json
{
  "data_info": {
    "total_records": 2192000,
    "unique_products": 2000,
    "date_range": {
      "start": "2022-01-01",
      "end": "2024-12-31"
    }
  }
}
```

## 📈 Data Quality

### Modern Dataset Features:
- ✅ **Current Patterns**: Post-COVID consumer behavior
- ✅ **Modern Events**: Black Friday, Prime Day, etc.
- ✅ **Realistic Seasonality**: Category-specific patterns
- ✅ **Supply Chain Effects**: 2022-2024 disruptions
- ✅ **Inflation Impact**: Economic conditions

### File Sizes:
- Modern dataset: ~110MB
- Legacy dataset: ~5MB
- Model files: ~1-5MB each (auto-generated)

## 🔧 Troubleshooting

**Issue**: "No processed data found"
**Solution**: Run `python create_modern_data.py` to generate the dataset

**Issue**: Service shows "legacy data" in logs
**Solution**: Ensure `daily_demand_by_product_modern.csv` exists in `data/processed/`

**Issue**: Predictions seem unrealistic
**Solution**: Verify you're using modern data (2022-2024) not legacy data (2010-2011)
