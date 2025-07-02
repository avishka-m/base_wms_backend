# #!/usr/bin/env python3
# """
# Test seasonal predictions with REAL processed e-commerce data
# """

# import sys
# import pandas as pd
# from pathlib import Path

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from models.prophet_forecaster import ProphetForecaster
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def test_real_data_predictions():
#     """Test predictions with real processed e-commerce data"""
#     print("🔮 TESTING WITH REAL E-COMMERCE DATA")
#     print("=" * 60)
    
#     # Load the real processed data
#     data_file = Path("data/processed/daily_demand_by_product.csv")
#     if not data_file.exists():
#         print("❌ Processed data not found. Run: python process_datasets.py")
#         return
    
#     print(f"📊 Loading real data from: {data_file}")
#     df = pd.read_csv(data_file)
    
#     print(f"✅ Loaded {len(df):,} records")
#     print(f"📅 Date range: {df['ds'].min()} to {df['ds'].max()}")
#     print(f"🏷️ Unique products: {df['product_id'].nunique():,}")
    
#     # Get top 5 products by total demand
#     top_products = df.groupby('product_id')['y'].sum().nlargest(5)
#     print(f"\n🏆 TOP 5 PRODUCTS TO FORECAST:")
#     for i, (product_id, total_demand) in enumerate(top_products.items(), 1):
#         product_info = df[df['product_id'] == product_id].iloc[0]
#         print(f"   {i}. {product_id}: {total_demand:,.0f} units - {product_info.get('Description', 'N/A')}")
    
#     print(f"\n🔮 GENERATING FORECASTS FOR TOP PRODUCTS...")
#     print("=" * 60)
    
#     # Test forecasting for each top product
#     results = {}
    
#     for i, (product_id, total_demand) in enumerate(top_products.items(), 1):
#         print(f"\n📋 FORECASTING PRODUCT {i}/5: {product_id}")
#         print("-" * 50)
        
#         # Get product data
#         product_data = df[df['product_id'] == product_id].copy()
#         product_data['ds'] = pd.to_datetime(product_data['ds'])
#         product_data = product_data.sort_values('ds')
        
#         print(f"📊 Product data: {len(product_data)} days of history")
#         print(f"📈 Total historical demand: {product_data['y'].sum():,.0f} units")
#         print(f"📅 Data range: {product_data['ds'].min().date()} to {product_data['ds'].max().date()}")
        
#         try:
#             # Create forecaster for this specific product
#             forecaster = ProphetForecaster(product_id=product_id)
            
#             # Train the model
#             print("🚀 Training Prophet model...")
#             cv_results = forecaster.train(product_data, 'y')
            
#             if 'error' not in cv_results:
#                 print("✅ Model trained successfully!")
                
#                 # Generate 90-day forecast
#                 print("🔮 Generating 90-day forecast...")
#                 forecast = forecaster.predict(periods=90)
                
#                 # Get forecast summary
#                 summary = forecaster.get_forecast_summary()
                
#                 # Display results
#                 print(f"\n📊 FORECAST RESULTS:")
#                 print(f"   🎯 Forecast period: {summary['forecast_periods']} days")
#                 print(f"   📈 Predicted total demand: {summary['total_predicted_demand']:,.0f} units")
#                 print(f"   📊 Average daily demand: {summary['mean_prediction']:.1f} units/day")
#                 print(f"   📉 Min predicted: {summary['min_prediction']:.1f} units/day")
#                 print(f"   📈 Max predicted: {summary['max_prediction']:.1f} units/day")
                
#                 # Save model
#                 model_path = forecaster.save_model()
#                 print(f"💾 Model saved: {Path(model_path).name}")
                
#                 # Store results
#                 results[product_id] = {
#                     'summary': summary,
#                     'historical_total': total_demand,
#                     'model_path': model_path,
#                     'status': 'success'
#                 }
                
#                 # Business insights
#                 avg_historical = product_data['y'].mean()
#                 trend = "increasing" if summary['mean_prediction'] > avg_historical else "decreasing"
#                 change_pct = ((summary['mean_prediction'] - avg_historical) / avg_historical) * 100
                
#                 print(f"\n💡 BUSINESS INSIGHTS:")
#                 print(f"   📊 Historical avg: {avg_historical:.1f} units/day")
#                 print(f"   🔮 Forecast avg: {summary['mean_prediction']:.1f} units/day")
#                 print(f"   📈 Trend: {trend} ({change_pct:+.1f}%)")
                
#                 if abs(change_pct) > 20:
#                     if change_pct > 0:
#                         print("   🔥 ALERT: Strong growth predicted - consider increasing stock!")
#                     else:
#                         print("   📉 ALERT: Declining demand - consider promotions!")
                
#             else:
#                 print(f"❌ Training failed: {cv_results['error']}")
#                 results[product_id] = {'status': 'failed', 'error': cv_results['error']}
                
#         except Exception as e:
#             print(f"❌ Error forecasting {product_id}: {e}")
#             results[product_id] = {'status': 'error', 'error': str(e)}
    
#     # Summary
#     print(f"\n🎉 FORECASTING SUMMARY")
#     print("=" * 40)
    
#     successful = [k for k, v in results.items() if v.get('status') == 'success']
#     failed = [k for k, v in results.items() if v.get('status') != 'success']
    
#     print(f"✅ Successful forecasts: {len(successful)}/{len(results)}")
#     print(f"❌ Failed forecasts: {len(failed)}")
    
#     if successful:
#         print(f"\n📊 FORECAST INSIGHTS:")
#         total_historical = sum(results[pid]['historical_total'] for pid in successful)
#         total_predicted = sum(results[pid]['summary']['total_predicted_demand'] for pid in successful)
        
#         print(f"   📈 Total historical (last year): {total_historical:,.0f} units")
#         print(f"   🔮 Total predicted (next 90 days): {total_predicted:,.0f} units")
#         print(f"   📊 Annualized prediction: {total_predicted * 4:,.0f} units/year")
        
#         growth = ((total_predicted * 4) - total_historical) / total_historical * 100
#         print(f"   📈 Projected growth: {growth:+.1f}%")
    
#     print(f"\n🚀 Your seasonal inventory system is ready for production!")
#     print("🔗 Next steps:")
#     print("   1. Connect to your MongoDB database")
#     print("   2. Update config.py with your settings") 
#     print("   3. Start the API: python -m backend.app.main")
#     print("   4. Integrate with your frontend")

# if __name__ == "__main__":
#     test_real_data_predictions()
