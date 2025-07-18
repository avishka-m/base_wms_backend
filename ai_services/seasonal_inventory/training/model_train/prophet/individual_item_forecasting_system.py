#!/usr/bin/env python3
"""
INDIVIDUAL ITEM FORECASTING SYSTEM
==================================
6 Category-Specific Models for Individual Product Demand Forecasting
Uses best parameters identified from category-level optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

class CategoryForecaster:
    """Individual forecaster for each category with optimized parameters"""
    
    def __init__(self, category_name):
        self.category_name = category_name
        self.best_params = self._get_category_best_params()
        self.holidays_df = self._create_holiday_calendar()
        self.trained_models = {}
        self.model_performance = {}
        
    def _get_category_best_params(self):
        """Get the best parameters for each category based on previous optimization"""
        
        # Best parameters identified from category-level analysis
        category_configs = {
            'books_media': {
                'changepoint_prior_scale': 0.5,
                'seasonality_mode': 'additive',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            'clothing': {
                'changepoint_prior_scale': 1.0,
                'seasonality_mode': 'additive',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            'electronics': {
                'changepoint_prior_scale': 0.5,
                'seasonality_mode': 'multiplicative',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            'health_beauty': {
                'changepoint_prior_scale': 0.01,
                'seasonality_mode': 'additive',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            'home_garden': {
                'changepoint_prior_scale': 1.0,
                'seasonality_mode': 'additive',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            'sports_outdoors': {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 0.1,
                'seasonality_mode': 'additive',
                'holidays_prior_scale': 10,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'daily_seasonality': False
            }
        }
        
        return category_configs.get(self.category_name, category_configs['books_media'])
    
    def _create_holiday_calendar(self):
        """Create holiday calendar for the category"""
        
        major_holidays = pd.DataFrame([
            # Black Friday
            {'holiday': 'black_friday', 'ds': '2022-11-25', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2023-11-24', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2024-11-29', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2025-11-28', 'lower_window': -1, 'upper_window': 1},
            
            # Christmas season
            {'holiday': 'christmas_season', 'ds': '2022-12-24', 'lower_window': -3, 'upper_window': 2},
            {'holiday': 'christmas_season', 'ds': '2023-12-24', 'lower_window': -3, 'upper_window': 2},
            {'holiday': 'christmas_season', 'ds': '2024-12-24', 'lower_window': -3, 'upper_window': 2},
            {'holiday': 'christmas_season', 'ds': '2025-12-24', 'lower_window': -3, 'upper_window': 2},
            
            # Valentine's Day
            {'holiday': 'valentine_day', 'ds': '2022-02-14', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'valentine_day', 'ds': '2023-02-14', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'valentine_day', 'ds': '2024-02-14', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'valentine_day', 'ds': '2025-02-14', 'lower_window': -1, 'upper_window': 1},
            
            # Back to School
            {'holiday': 'back_to_school', 'ds': '2022-08-15', 'lower_window': -5, 'upper_window': 5},
            {'holiday': 'back_to_school', 'ds': '2023-08-15', 'lower_window': -5, 'upper_window': 5},
            {'holiday': 'back_to_school', 'ds': '2024-08-15', 'lower_window': -5, 'upper_window': 5},
            {'holiday': 'back_to_school', 'ds': '2025-08-15', 'lower_window': -5, 'upper_window': 5},
        ])
        
        major_holidays['ds'] = pd.to_datetime(major_holidays['ds'])
        return major_holidays
    
    def _add_enhanced_regressors(self, df):
        """Add enhanced regressors based on category analysis"""
        
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Weekend indicator
        df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Month-end effect (last 5 days of month)
        df['is_month_end'] = (df['ds'].dt.day >= df['ds'].dt.days_in_month - 4).astype(int)
        
        # Payday effect (1st and 15th of month, plus weekends)
        payday_days = [1, 15]
        df['is_payday'] = df['ds'].dt.day.isin(payday_days).astype(int)
        
        # Summer surge (June-August)
        df['is_summer'] = df['ds'].dt.month.isin([6, 7, 8]).astype(int)
        
        # Year-end shopping (November-December)
        df['is_year_end'] = df['ds'].dt.month.isin([11, 12]).astype(int)
        
        return df
    
    def train_individual_model(self, product_data, product_id, min_data_points=365):
        """Train model for individual product within the category"""
        
        if len(product_data) < min_data_points:
            print(f"  ⚠️  Skipping {product_id}: insufficient data ({len(product_data)} < {min_data_points})")
            return None, None
        
        try:
            # Prepare data with regressors
            enhanced_data = self._add_enhanced_regressors(product_data)
            
            # Create Prophet model with category-specific parameters
            model = Prophet(
                holidays=self.holidays_df,
                **self.best_params
            )
            
            # Add regressors based on category
            regressors = ['is_weekend', 'is_month_end', 'is_payday']
            
            # Category-specific regressors
            if self.category_name in ['books_media', 'home_garden']:
                regressors.append('is_summer')
            if self.category_name in ['clothing', 'electronics']:
                regressors.append('is_year_end')
            
            for regressor in regressors:
                model.add_regressor(regressor, prior_scale=10, standardize=False)
            
            # Train model
            train_cols = ['ds', 'y'] + regressors
            model.fit(enhanced_data[train_cols])
            
            # Calculate training performance (last 20% as validation)
            val_size = max(30, int(len(enhanced_data) * 0.2))
            train_data = enhanced_data[:-val_size]
            val_data = enhanced_data[-val_size:]
            
            if len(val_data) > 0:
                val_forecast = model.predict(val_data[['ds'] + regressors])
                val_rmse = np.sqrt(mean_squared_error(val_data['y'], val_forecast['yhat']))
                val_mae = mean_absolute_error(val_data['y'], val_forecast['yhat'])
                val_r2 = r2_score(val_data['y'], val_forecast['yhat'])
            else:
                val_rmse = val_mae = val_r2 = 0
            
            performance = {
                'rmse': val_rmse,
                'mae': val_mae,
                'r2': val_r2,
                'data_points': len(enhanced_data),
                'avg_demand': enhanced_data['y'].mean(),
                'regressors': regressors
            }
            
            return model, performance
            
        except Exception as e:
            print(f"  ❌ Error training {product_id}: {str(e)}")
            return None, None
    
    def forecast_individual_product(self, model, last_data_point, forecast_days=30):
        """Generate forecast for individual product"""
        
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Add regressors to future dates
            future_enhanced = self._add_enhanced_regressors(future)
            
            # Get regressor columns used in training
            regressor_cols = []
            for regressor in ['is_weekend', 'is_month_end', 'is_payday', 'is_summer', 'is_year_end']:
                if regressor in future_enhanced.columns:
                    regressor_cols.append(regressor)
            
            # Generate forecast
            forecast = model.predict(future_enhanced[['ds'] + regressor_cols])
            
            # Return only future predictions
            future_forecast = forecast.tail(forecast_days).copy()
            
            return future_forecast
            
        except Exception as e:
            print(f"  ❌ Forecast error: {str(e)}")
            return None
    
    def save_model(self, model, product_id, save_dir='individual_models'):
        """Save trained model to disk"""
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        model_path = os.path.join(save_dir, f'{self.category_name}_{product_id}_model.pkl')
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            return model_path
        except Exception as e:
            print(f"  ⚠️  Failed to save model for {product_id}: {str(e)}")
            return None
    
    def load_model(self, product_id, save_dir='individual_models'):
        """Load trained model from disk"""
        
        model_path = os.path.join(save_dir, f'{self.category_name}_{product_id}_model.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"  ⚠️  Failed to load model for {product_id}: {str(e)}")
            return None

class IndividualItemForecastingSystem:
    """Main system for managing individual item forecasting across all categories"""
    
    def __init__(self):
        self.category_forecasters = {}
        self.data = None
        self.results = {}
        
    def load_data(self, file_path='daily_demand_by_product_modern.csv'):
        """Load and prepare data for individual item forecasting"""
        
        print("Loading data for individual item forecasting...")
        
        self.data = pd.read_csv(file_path)
        self.data['ds'] = pd.to_datetime(self.data['ds'])
        
        print(f"Total data points: {len(self.data):,}")
        print(f"Date range: {self.data['ds'].min()} to {self.data['ds'].max()}")
        print(f"Categories: {self.data['category'].unique()}")
        print(f"Unique products: {self.data['product_id'].nunique():,}")
        
        return self.data
    
    def initialize_category_forecasters(self):
        """Initialize forecasters for each category"""
        
        categories = self.data['category'].unique()
        
        for category in categories:
            self.category_forecasters[category] = CategoryForecaster(category)
            print(f"✅ Initialized forecaster for {category}")
    
    def train_category_models(self, category, max_products_per_category=50, parallel=True):
        """Train models for all products in a category"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING INDIVIDUAL MODELS: {category.upper()}")
        print(f"{'='*60}")
        
        # Get category data
        category_data = self.data[self.data['category'] == category].copy()
        
        # Get products with sufficient data
        product_counts = category_data.groupby('product_id').size()
        valid_products = product_counts[product_counts >= 365].index[:max_products_per_category]
        
        print(f"Products with sufficient data: {len(valid_products)}")
        print(f"Training models for: {min(len(valid_products), max_products_per_category)} products")
        
        forecaster = self.category_forecasters[category]
        
        # Training function for parallel processing
        def train_single_product(product_id):
            product_data = category_data[category_data['product_id'] == product_id].copy()
            product_data = product_data.sort_values('ds').reset_index(drop=True)
            
            model, performance = forecaster.train_individual_model(product_data, product_id)
            
            if model is not None:
                # Save model
                model_path = forecaster.save_model(model, product_id)
                return product_id, model, performance, model_path
            
            return product_id, None, None, None
        
        # Train models
        trained_models = {}
        model_performance = {}
        
        if parallel and len(valid_products) > 5:
            # Parallel training for larger datasets
            print("🔄 Training models in parallel...")
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_product = {executor.submit(train_single_product, pid): pid for pid in valid_products}
                
                for i, future in enumerate(as_completed(future_to_product), 1):
                    product_id = future_to_product[future]
                    
                    try:
                        pid, model, performance, model_path = future.result()
                        
                        if model is not None:
                            trained_models[pid] = {
                                'model': model,
                                'model_path': model_path
                            }
                            model_performance[pid] = performance
                            
                            print(f"  ✅ [{i:3d}/{len(valid_products)}] {pid}: "
                                  f"RMSE={performance['rmse']:.1f}, "
                                  f"R²={performance['r2']:.3f}, "
                                  f"Points={performance['data_points']}")
                        
                    except Exception as e:
                        print(f"  ❌ [{i:3d}/{len(valid_products)}] {product_id}: Error - {str(e)}")
        else:
            # Sequential training
            print("🔄 Training models sequentially...")
            
            for i, product_id in enumerate(valid_products, 1):
                try:
                    pid, model, performance, model_path = train_single_product(product_id)
                    
                    if model is not None:
                        trained_models[pid] = {
                            'model': model,
                            'model_path': model_path
                        }
                        model_performance[pid] = performance
                        
                        print(f"  ✅ [{i:3d}/{len(valid_products)}] {pid}: "
                              f"RMSE={performance['rmse']:.1f}, "
                              f"R²={performance['r2']:.3f}, "
                              f"Points={performance['data_points']}")
                    
                except Exception as e:
                    print(f"  ❌ [{i:3d}/{len(valid_products)}] {product_id}: Error - {str(e)}")
        
        # Store results
        self.results[category] = {
            'trained_models': trained_models,
            'model_performance': model_performance,
            'forecaster': forecaster
        }
        
        # Summary
        print(f"\n📊 TRAINING SUMMARY - {category.upper()}:")
        print(f"  Models trained: {len(trained_models)}")
        print(f"  Avg RMSE: {np.mean([p['rmse'] for p in model_performance.values()]):.1f}")
        print(f"  Avg R²: {np.mean([p['r2'] for p in model_performance.values()]):.3f}")
        print(f"  Avg data points: {np.mean([p['data_points'] for p in model_performance.values()]):.0f}")
        
        return trained_models, model_performance
    
    def forecast_category_products(self, category, forecast_days=30):
        """Generate forecasts for all products in a category"""
        
        print(f"\n🔮 GENERATING FORECASTS: {category.upper()}")
        print(f"Forecast horizon: {forecast_days} days")
        
        if category not in self.results:
            print(f"❌ No trained models found for {category}")
            return None
        
        forecaster = self.results[category]['forecaster']
        trained_models = self.results[category]['trained_models']
        
        forecasts = {}
        
        for product_id, model_info in trained_models.items():
            try:
                model = model_info['model']
                
                # Get last data point for context
                last_data = self.data[self.data['product_id'] == product_id]['ds'].max()
                
                # Generate forecast
                forecast = forecaster.forecast_individual_product(model, last_data, forecast_days)
                
                if forecast is not None:
                    forecasts[product_id] = forecast
                    print(f"  ✅ {product_id}: Forecast generated")
                
            except Exception as e:
                print(f"  ❌ {product_id}: Forecast error - {str(e)}")
        
        print(f"📈 Generated forecasts for {len(forecasts)} products")
        
        return forecasts
    
    def create_summary_report(self):
        """Create comprehensive summary report"""
        
        print(f"\n{'='*80}")
        print("INDIVIDUAL ITEM FORECASTING SYSTEM - SUMMARY REPORT")
        print(f"{'='*80}")
        
        total_models = 0
        total_products = 0
        
        for category, results in self.results.items():
            trained_models = results['trained_models']
            performance = results['model_performance']
            
            models_count = len(trained_models)
            total_models += models_count
            
            if models_count > 0:
                avg_rmse = np.mean([p['rmse'] for p in performance.values()])
                avg_r2 = np.mean([p['r2'] for p in performance.values()])
                avg_data_points = np.mean([p['data_points'] for p in performance.values()])
                
                print(f"\n📊 {category.upper()}:")
                print(f"  Models trained: {models_count}")
                print(f"  Average RMSE: {avg_rmse:.1f}")
                print(f"  Average R²: {avg_r2:.3f}")
                print(f"  Average data points: {avg_data_points:.0f}")
                
                # Best and worst performers
                if len(performance) > 0:
                    best_product = min(performance.keys(), key=lambda x: performance[x]['rmse'])
                    worst_product = max(performance.keys(), key=lambda x: performance[x]['rmse'])
                    
                    print(f"  Best performer: {best_product} (RMSE: {performance[best_product]['rmse']:.1f})")
                    print(f"  Needs attention: {worst_product} (RMSE: {performance[worst_product]['rmse']:.1f})")
        
        print(f"\n🎯 OVERALL SUMMARY:")
        print(f"  Total models trained: {total_models}")
        print(f"  Categories covered: {len(self.results)}")
        print(f"  System status: {'✅ Ready for production' if total_models > 0 else '❌ No models trained'}")
        
        return self.results
    
    def forecast_specific_product(self, product_id, forecast_days=30):
        """Generate forecast for a specific product"""
        
        # Find product category
        product_data = self.data[self.data['product_id'] == product_id]
        
        if len(product_data) == 0:
            print(f"❌ Product {product_id} not found")
            return None
        
        category = product_data['category'].iloc[0]
        
        if category not in self.results:
            print(f"❌ No models trained for category {category}")
            return None
        
        if product_id not in self.results[category]['trained_models']:
            print(f"❌ No model trained for product {product_id}")
            return None
        
        print(f"🔮 Generating forecast for {product_id} ({category})...")
        
        forecaster = self.results[category]['forecaster']
        model = self.results[category]['trained_models'][product_id]['model']
        
        forecast = forecaster.forecast_individual_product(model, product_data['ds'].max(), forecast_days)
        
        if forecast is not None:
            print(f"✅ Forecast generated for {forecast_days} days")
            print(f"   Forecast range: {forecast['yhat'].min():.0f} - {forecast['yhat'].max():.0f}")
            print(f"   Average forecast: {forecast['yhat'].mean():.0f}")
        
        return forecast

def main():
    """Main execution function"""
    
    print("="*80)
    print("INDIVIDUAL ITEM FORECASTING SYSTEM")
    print("6 Category-Specific Models for Individual Product Demand")
    print("="*80)
    
    # Initialize system
    forecasting_system = IndividualItemForecastingSystem()
    
    # Load data
    data = forecasting_system.load_data()
    
    # Initialize category forecasters
    forecasting_system.initialize_category_forecasters()
    
    # Train models for each category
    categories = data['category'].unique()
    
    for category in categories:
        try:
            forecasting_system.train_category_models(category, max_products_per_category=20, parallel=True)
        except Exception as e:
            print(f"❌ Error training {category}: {str(e)}")
    
    # Create summary report
    results = forecasting_system.create_summary_report()
    
    # Example: Generate forecasts for a category
    if len(results) > 0:
        example_category = list(results.keys())[0]
        print(f"\n🔮 Generating example forecasts for {example_category}...")
        forecasts = forecasting_system.forecast_category_products(example_category, forecast_days=30)
    
    print(f"\n{'='*80}")
    print("✅ INDIVIDUAL ITEM FORECASTING SYSTEM READY!")
    print("✅ Use forecast_specific_product(product_id) for individual forecasts")
    print("✅ Use forecast_category_products(category) for bulk forecasts")
    print("="*80)
    
    return forecasting_system

if __name__ == "__main__":
    # Execute the individual item forecasting system
    system = main()
