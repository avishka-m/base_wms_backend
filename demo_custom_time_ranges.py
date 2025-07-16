#!/usr/bin/env python3
"""
Demo: Custom Time Range Forecasting with Prophet System
Shows how to predict demand for different time ranges and start dates
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def demo_custom_time_ranges():
    """Demonstrate custom time range forecasting"""
    
    print("🎯 CUSTOM TIME RANGE FORECASTING DEMO")
    print("=" * 50)
    
    # Note: This is the structure - you'll need to fix import issues first
    # from app.services.prophet_forecasting_service import ProphetForecastingService
    # service = ProphetForecastingService()
    
    # For now, let's show the usage patterns:
    
    print("\n📋 AVAILABLE TIME RANGE OPTIONS:")
    print("-" * 30)
    
    # Example configurations
    time_range_examples = [
        {
            "name": "1 Week Forecast",
            "horizon_days": 7,
            "start_date": None,  # Tomorrow
            "description": "Short-term forecast starting tomorrow"
        },
        {
            "name": "1 Month Forecast", 
            "horizon_days": 30,
            "start_date": None,
            "description": "Standard monthly forecast"
        },
        {
            "name": "3 Month Forecast",
            "horizon_days": 90, 
            "start_date": None,
            "description": "Quarterly planning forecast"
        },
        {
            "name": "6 Month Forecast",
            "horizon_days": 180,
            "start_date": None,
            "description": "Long-term strategic forecast" 
        },
        {
            "name": "Custom Start Date",
            "horizon_days": 30,
            "start_date": "2025-08-01",  # Specific start date
            "description": "30-day forecast starting August 1st"
        },
        {
            "name": "Holiday Season",
            "horizon_days": 60,
            "start_date": "2025-11-01",  # Pre-holiday forecast
            "description": "Holiday season forecast (Nov-Dec)"
        },
        {
            "name": "Next Year Planning",
            "horizon_days": 365,
            "start_date": "2026-01-01",  # Next year
            "description": "Full year forecast for 2026"
        }
    ]
    
    for i, config in enumerate(time_range_examples, 1):
        print(f"{i}. {config['name']}")
        print(f"   📅 Duration: {config['horizon_days']} days")
        print(f"   🚀 Start: {config['start_date'] or 'Tomorrow'}")
        print(f"   📝 Use Case: {config['description']}")
        print()
    
    print("\n🔧 CODE EXAMPLES:")
    print("-" * 30)
    
    print("# 1. One Week Forecast (starting tomorrow)")
    print("result = await service.predict_item_demand(")
    print("    item_id='PROD123',")
    print("    horizon_days=7")
    print(")")
    print()
    
    print("# 2. Custom Start Date (30 days from August 1st)")
    print("result = await service.predict_item_demand(")
    print("    item_id='PROD123',") 
    print("    horizon_days=30,")
    print("    start_date='2025-08-01'")
    print(")")
    print()
    
    print("# 3. Long-term with custom confidence")
    print("result = await service.predict_item_demand(")
    print("    item_id='PROD123',")
    print("    horizon_days=180,  # 6 months")
    print("    confidence_interval=0.90  # 90% confidence")
    print(")")
    print()
    
    print("# 4. Holiday Season Planning")
    print("result = await service.predict_item_demand(")
    print("    item_id='PROD123',")
    print("    horizon_days=60,")
    print("    start_date='2025-11-01',  # Start Nov 1st")
    print("    confidence_interval=0.95")
    print(")")
    print()

async def demo_batch_time_ranges():
    """Show batch forecasting with different time ranges"""
    
    print("\n🎯 BATCH FORECASTING WITH CUSTOM RANGES")
    print("=" * 50)
    
    print("\n📦 Example: Multiple Products, Multiple Time Ranges")
    print("-" * 30)
    
    # Example batch configuration
    products = ["PROD001", "PROD002", "PROD003"]
    time_ranges = [
        {"days": 7, "name": "1 Week"},
        {"days": 30, "name": "1 Month"}, 
        {"days": 90, "name": "3 Months"}
    ]
    
    print("# Batch forecast example")
    print("products = ['PROD001', 'PROD002', 'PROD003']")
    print("time_ranges = [7, 30, 90]  # days")
    print()
    print("for product in products:")
    print("    for days in time_ranges:")
    print("        result = await service.predict_item_demand(")
    print("            item_id=product,")
    print("            horizon_days=days")
    print("        )")
    print("        print(f'{product}: {days}-day forecast complete')")
    print()
    
    # Show expected results structure
    print("\n📊 EXPECTED RESULT STRUCTURE:")
    print("-" * 30)
    
    result_example = {
        "status": "success",
        "item_id": "PROD123", 
        "category": "electronics",
        "forecast_period": {
            "start_date": "2025-07-17",
            "end_date": "2025-08-15", 
            "days": 30
        },
        "predictions": [
            {
                "date": "2025-07-17",
                "predicted_demand": 45.67,
                "lower_bound": 38.23,
                "upper_bound": 53.11,
                "confidence": 0.95,
                "category": "electronics",
                "category_share": 12.5
            },
            "... more daily predictions ..."
        ],
        "summary": {
            "total_predicted_demand": 1370.10,
            "average_daily_demand": 45.67,
            "peak_daily_demand": 67.89,
            "confidence_interval": 0.95
        },
        "model_info": {
            "model_type": "Prophet-Category",
            "category_model": "electronics", 
            "hyperparameters": {
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 1.0,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0
            }
        }
    }
    
    print("Result structure includes:")
    print("✅ Daily predictions for each day in range")
    print("✅ Forecast period (start/end dates)")
    print("✅ Summary statistics (total, average, peak demand)")
    print("✅ Confidence intervals (upper/lower bounds)")
    print("✅ Category information and model details")
    print("✅ Your optimized hyperparameters used")

def demo_time_range_calculator():
    """Helper to calculate different time ranges"""
    
    print("\n⏰ TIME RANGE CALCULATOR")
    print("=" * 50)
    
    now = datetime.now()
    
    ranges = [
        ("1 Week", 7),
        ("2 Weeks", 14), 
        ("1 Month", 30),
        ("1 Quarter", 90),
        ("6 Months", 180),
        ("1 Year", 365)
    ]
    
    print(f"📅 Starting from: {now.strftime('%Y-%m-%d')}")
    print()
    
    for name, days in ranges:
        end_date = now + timedelta(days=days)
        print(f"{name:10} ({days:3d} days): {now.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    
    print()
    print("🎯 SPECIAL DATE RANGES:")
    print("-" * 30)
    
    # Special business periods
    special_ranges = [
        ("Q4 2025", "2025-10-01", 92),  # Q4
        ("Holiday Season", "2025-11-15", 45),  # Nov 15 - Dec 30
        ("New Year 2026", "2026-01-01", 31),  # January 2026
        ("Summer 2025", "2025-06-01", 92),     # Summer quarter
    ]
    
    for name, start_str, days in special_ranges:
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = start_date + timedelta(days=days)
        print(f"{name:15}: {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')} ({days} days)")

if __name__ == "__main__":
    print("🚀 Prophet Forecasting - Custom Time Range Demo")
    print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run demos
    asyncio.run(demo_custom_time_ranges())
    asyncio.run(demo_batch_time_ranges()) 
    demo_time_range_calculator()
    
    print("\n" + "=" * 50)
    print("✅ Demo Complete!")
    print("💡 Your system supports ANY custom time range!")
    print("=" * 50)
