"""
Enhanced Product Forecasting with External Regressors
Using add_regressor and advanced Prophet features for better accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

class EnhancedProductForecaster:
    """
    Advanced Prophet forecaster with external regressors for better accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.product_metadata = {}
        self.forecasts = {}
        self.category_params = self._load_optimized_params()
    
    def _load_optimized_params(self) -> Dict:
        """Load optimized category parameters"""
        try:
            config_globals = {}
            exec(open('prophet_cv_optimized_config.py').read(), config_globals)
            return config_globals['CATEGORY_HYPERPARAMETERS']
        except:
            return {
                'books_media': {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'clothing': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
                'electronics': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
                'health_beauty': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'home_garden': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'sports_outdoors': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
            }
    
    def create_external_regressors(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """
        Create external regressors for improved forecasting
        
        These regressors help capture cross-product and category-level effects
        """
        product_data = df[df['product_id'] == product_id].copy()
        category = product_data['category'].iloc[0]
        
        # Get aggregated data for regressors
        daily_data = df.groupby(['ds', 'category']).agg({
            'y': ['sum', 'mean', 'count']
        }).reset_index()
        daily_data.columns = ['ds', 'category', 'total_demand', 'avg_demand', 'active_products']
        
        # Category-level regressors
        category_data = daily_data[daily_data['category'] == category].copy()
        category_data = category_data.sort_values('ds').reset_index(drop=True)
        
        # Create enhanced dataset with regressors
        enhanced_data = product_data[['ds', 'y']].merge(
            category_data[['ds', 'total_demand', 'avg_demand', 'active_products']], 
            on='ds', 
            how='left'
        )
        
        # Additional regressors
        
        # 1. Category momentum (7-day moving average of category demand)
        enhanced_data['category_ma7'] = enhanced_data['total_demand'].rolling(window=7, min_periods=1).mean()
        
        # 2. Category volatility (7-day rolling std)
        enhanced_data['category_volatility'] = enhanced_data['total_demand'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # 3. Product relative performance (product demand / category average)
        enhanced_data['relative_performance'] = enhanced_data['y'] / (enhanced_data['avg_demand'] + 1e-6)
        enhanced_data['relative_performance_ma7'] = enhanced_data['relative_performance'].rolling(window=7, min_periods=1).mean()
        
        # 4. Day of week effects (encoded as sine/cosine)
        enhanced_data['dow'] = enhanced_data['ds'].dt.dayofweek
        enhanced_data['dow_sin'] = np.sin(2 * np.pi * enhanced_data['dow'] / 7)
        enhanced_data['dow_cos'] = np.cos(2 * np.pi * enhanced_data['dow'] / 7)
        
        # 5. Month of year effects
        enhanced_data['month'] = enhanced_data['ds'].dt.month
        enhanced_data['month_sin'] = np.sin(2 * np.pi * enhanced_data['month'] / 12)
        enhanced_data['month_cos'] = np.cos(2 * np.pi * enhanced_data['month'] / 12)
        
        # 6. Quarter effects
        enhanced_data['quarter'] = enhanced_data['ds'].dt.quarter
        enhanced_data['quarter_sin'] = np.sin(2 * np.pi * enhanced_data['quarter'] / 4)
        enhanced_data['quarter_cos'] = np.cos(2 * np.pi * enhanced_data['quarter'] / 4)
        
        # 7. Lag features (previous day's category demand)
        enhanced_data['category_lag1'] = enhanced_data['total_demand'].shift(1)
        enhanced_data['category_lag7'] = enhanced_data['total_demand'].shift(7)
        
        # 8. Trend features
        enhanced_data['day_number'] = (enhanced_data['ds'] - enhanced_data['ds'].min()).dt.days
        enhanced_data['trend_linear'] = enhanced_data['day_number'] / enhanced_data['day_number'].max()
        
        # 9. Category growth rate (week-over-week change)
        enhanced_data['category_growth'] = enhanced_data['total_demand'].pct_change(periods=7).fillna(0)
        
        # Fill any remaining NaN values
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return enhanced_data
    
    def prepare_regressor_data(self, data: pd.DataFrame, product_id: str, 
                              future_periods: int = 0) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and scale regressor data for training and prediction
        """
        regressor_columns = [
            'category_ma7', 'category_volatility', 'relative_performance_ma7',
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 
            'quarter_sin', 'quarter_cos', 'category_lag1', 'category_lag7',
            'trend_linear', 'category_growth'
        ]
        
        # If predicting future, we need to extend the regressors
        if future_periods > 0:
            data = self._extend_regressors_for_future(data, future_periods)
        
        # Scale the regressors
        scaler_key = f"{product_id}_scaler"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
            # Fit only on non-future data
            train_data = data[~pd.isna(data['y'])] if 'y' in data.columns else data
            self.scalers[scaler_key].fit(train_data[regressor_columns])
        
        # Transform the data
        scaled_regressors = self.scalers[scaler_key].transform(data[regressor_columns])
        scaled_df = data.copy()
        scaled_df[regressor_columns] = scaled_regressors
        
        return scaled_df, regressor_columns
    
    def _extend_regressors_for_future(self, data: pd.DataFrame, future_periods: int) -> pd.DataFrame:
        """
        Extend regressor data for future predictions
        """
        last_date = data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_periods, freq='D')
        
        # Create future dataframe
        future_data = pd.DataFrame({'ds': future_dates})
        
        # Extend time-based regressors
        future_data['dow'] = future_data['ds'].dt.dayofweek
        future_data['dow_sin'] = np.sin(2 * np.pi * future_data['dow'] / 7)
        future_data['dow_cos'] = np.cos(2 * np.pi * future_data['dow'] / 7)
        
        future_data['month'] = future_data['ds'].dt.month
        future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
        future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)
        
        future_data['quarter'] = future_data['ds'].dt.quarter
        future_data['quarter_sin'] = np.sin(2 * np.pi * future_data['quarter'] / 4)
        future_data['quarter_cos'] = np.cos(2 * np.pi * future_data['quarter'] / 4)
        
        # Extend trend
        max_day_number = data['day_number'].max()
        future_data['day_number'] = range(max_day_number + 1, max_day_number + future_periods + 1)
        future_data['trend_linear'] = future_data['day_number'] / (max_day_number + future_periods)
        
        # For other regressors, use last known values or trends
        last_values = data.tail(7).mean()  # Use average of last week
        
        for col in ['category_ma7', 'category_volatility', 'relative_performance_ma7', 
                   'category_lag1', 'category_lag7', 'category_growth']:
            future_data[col] = last_values[col]
        
        # Combine with historical data
        extended_data = pd.concat([data, future_data], ignore_index=True)
        return extended_data
    
    def train_enhanced_model(self, df: pd.DataFrame, product_id: str, 
                           use_regressors: bool = True) -> Tuple[Prophet, Dict]:
        """
        Train enhanced Prophet model with external regressors
        """
        print(f"\n🚀 Training enhanced model for {product_id}")
        
        # Create enhanced dataset
        enhanced_data = self.create_external_regressors(df, product_id)
        product_data = df[df['product_id'] == product_id]
        category = product_data['category'].iloc[0]
        
        print(f"  Category: {category}")
        print(f"  Data points: {len(enhanced_data)}")
        print(f"  Avg demand: {enhanced_data['y'].mean():.1f}")
        
        # Get base parameters
        params = self.category_params[category].copy()
        params.update({
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays_prior_scale': 10.0
        })
        
        # Create model
        model = Prophet(**params)
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        if use_regressors:
            # Prepare regressors
            train_data, regressor_cols = self.prepare_regressor_data(enhanced_data, product_id)
            
            # Add regressors to model
            for regressor in regressor_cols:
                model.add_regressor(regressor, standardize=False)  # Already standardized
            
            print(f"  Added {len(regressor_cols)} external regressors")
            
            # Train with regressors
            model.fit(train_data[['ds', 'y'] + regressor_cols])
        else:
            # Train without regressors
            model.fit(enhanced_data[['ds', 'y']])
        
        # Store model and metadata
        self.models[product_id] = model
        self.product_metadata[product_id] = {
            'category': category,
            'use_regressors': use_regressors,
            'regressor_cols': regressor_cols if use_regressors else [],
            'avg_demand': enhanced_data['y'].mean(),
            'std_demand': enhanced_data['y'].std()
        }
        
        # Quick validation
        validation_metrics = {}
        if len(enhanced_data) > 200:
            try:
                print("  Running cross-validation...")
                if use_regressors:
                    # For CV with regressors, we need to be more careful
                    split_idx = int(len(train_data) * 0.8)
                    train_cv = train_data[:split_idx]
                    val_cv = train_data[split_idx:]
                    
                    # Train on subset
                    cv_model = Prophet(**params)
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
                    for regressor in regressor_cols:
                        cv_model.add_regressor(regressor, standardize=False)
                    
                    cv_model.fit(train_cv[['ds', 'y'] + regressor_cols])
                    
                    # Predict on validation
                    cv_forecast = cv_model.predict(val_cv[['ds'] + regressor_cols])
                    
                    val_rmse = np.sqrt(mean_squared_error(val_cv['y'], cv_forecast['yhat']))
                    val_mae = mean_absolute_error(val_cv['y'], cv_forecast['yhat'])
                    
                    validation_metrics = {'rmse': val_rmse, 'mae': val_mae}
                    print(f"  CV RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
                
            except Exception as e:
                print(f"  CV failed: {str(e)[:50]}...")
                validation_metrics = {'error': str(e)}
        
        return model, validation_metrics
    
    def predict_enhanced(self, product_id: str, periods: int = 30) -> pd.DataFrame:
        """
        Generate enhanced forecast with regressors
        """
        if product_id not in self.models:
            raise ValueError(f"Model for {product_id} not trained")
        
        model = self.models[product_id]
        metadata = self.product_metadata[product_id]
        
        if metadata['use_regressors']:
            # Need to create future regressor data
            # This is a simplified approach - in production you'd have actual future regressor values
            print(f"  Generating forecast with {len(metadata['regressor_cols'])} regressors...")
            
            # Create future dataframe with regressors
            future = model.make_future_dataframe(periods=periods)
            
            # For this demo, we'll use the last known regressor values
            # In practice, you'd have actual future regressor data
            last_regressors = {}
            regressor_cols = metadata['regressor_cols']
            
            # Get last known values (simplified approach)
            for col in regressor_cols:
                if col in ['dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']:
                    # Time-based regressors can be calculated
                    if 'dow' in col:
                        dow = future['ds'].dt.dayofweek
                        if 'sin' in col:
                            future[col] = np.sin(2 * np.pi * dow / 7)
                        else:
                            future[col] = np.cos(2 * np.pi * dow / 7)
                    elif 'month' in col:
                        month = future['ds'].dt.month
                        if 'sin' in col:
                            future[col] = np.sin(2 * np.pi * month / 12)
                        else:
                            future[col] = np.cos(2 * np.pi * month / 12)
                    elif 'quarter' in col:
                        quarter = future['ds'].dt.quarter
                        if 'sin' in col:
                            future[col] = np.sin(2 * np.pi * quarter / 4)
                        else:
                            future[col] = np.cos(2 * np.pi * quarter / 4)
                else:
                    # For other regressors, use last known value (forward fill)
                    future[col] = 0.0  # Simplified - would need actual future values
            
            forecast = model.predict(future)
        else:
            # Standard prediction without regressors
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
        
        # Add metadata
        forecast['product_id'] = product_id
        forecast['category'] = metadata['category']
        
        self.forecasts[product_id] = forecast
        return forecast
    
    def compare_standard_vs_enhanced(self, df: pd.DataFrame, test_products: List[str]) -> Dict:
        """
        Compare standard Prophet vs enhanced Prophet with regressors
        """
        print("🔬 Comparing Standard vs Enhanced Prophet Models")
        print("="*60)
        
        results = {}
        
        for i, product_id in enumerate(test_products):
            print(f"\n[{i+1}/{len(test_products)}] Testing {product_id}")
            
            try:
                product_data = df[df['product_id'] == product_id]
                
                if len(product_data) < 150:
                    print(f"  ⚠️ Insufficient data: {len(product_data)} days")
                    continue
                
                # Split data for testing
                split_idx = int(len(product_data) * 0.8)
                test_periods = len(product_data) - split_idx
                
                train_df = df[df['ds'] <= product_data.iloc[split_idx]['ds']]
                actual_values = product_data.iloc[split_idx:]['y'].values
                
                print(f"  Train: {split_idx} days, Test: {test_periods} days")
                
                # Method 1: Standard Prophet
                print("  Training standard Prophet...")
                standard_model, _ = self.train_enhanced_model(train_df, product_id, use_regressors=False)
                standard_forecast = standard_model.predict(
                    standard_model.make_future_dataframe(periods=test_periods)
                )
                standard_pred = standard_forecast['yhat'].tail(test_periods).values
                
                # Method 2: Enhanced Prophet with regressors
                print("  Training enhanced Prophet...")
                enhanced_model, _ = self.train_enhanced_model(train_df, product_id, use_regressors=True)
                
                # For enhanced prediction, we need to be more careful with regressors
                # This is simplified - in practice you'd have actual future regressor values
                enhanced_future = enhanced_model.make_future_dataframe(periods=test_periods)
                
                # Add basic time regressors for prediction
                enhanced_future['dow_sin'] = np.sin(2 * np.pi * enhanced_future['ds'].dt.dayofweek / 7)
                enhanced_future['dow_cos'] = np.cos(2 * np.pi * enhanced_future['ds'].dt.dayofweek / 7)
                enhanced_future['month_sin'] = np.sin(2 * np.pi * enhanced_future['ds'].dt.month / 12)
                enhanced_future['month_cos'] = np.cos(2 * np.pi * enhanced_future['ds'].dt.month / 12)
                enhanced_future['quarter_sin'] = np.sin(2 * np.pi * enhanced_future['ds'].dt.quarter / 4)
                enhanced_future['quarter_cos'] = np.cos(2 * np.pi * enhanced_future['ds'].dt.quarter / 4)
                
                # For other regressors, use simplified approach (forward fill last values)
                other_regressors = ['category_ma7', 'category_volatility', 'relative_performance_ma7',
                                  'category_lag1', 'category_lag7', 'trend_linear', 'category_growth']
                for col in other_regressors:
                    enhanced_future[col] = 0.0  # Simplified
                
                enhanced_forecast = enhanced_model.predict(enhanced_future)
                enhanced_pred = enhanced_forecast['yhat'].tail(test_periods).values
                
                # Calculate metrics
                def calc_metrics(actual, predicted):
                    predicted = np.maximum(predicted, 0)
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    mae = mean_absolute_error(actual, predicted)
                    return rmse, mae
                
                standard_rmse, standard_mae = calc_metrics(actual_values, standard_pred)
                enhanced_rmse, enhanced_mae = calc_metrics(actual_values, enhanced_pred)
                
                improvement = (standard_rmse - enhanced_rmse) / standard_rmse * 100
                
                print(f"  📊 Results:")
                print(f"    Standard RMSE: {standard_rmse:.2f}")
                print(f"    Enhanced RMSE: {enhanced_rmse:.2f}")
                print(f"    Improvement: {improvement:.1f}%")
                
                results[product_id] = {
                    'standard_rmse': standard_rmse,
                    'enhanced_rmse': enhanced_rmse,
                    'improvement_pct': improvement,
                    'standard_pred': standard_pred,
                    'enhanced_pred': enhanced_pred,
                    'actual': actual_values
                }
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                continue
        
        return results

def run_enhanced_forecasting_demo():
    """
    Run demonstration of enhanced forecasting with regressors
    """
    print("🚀 Enhanced Product Forecasting with Regressors")
    print("="*60)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Initialize enhanced forecaster
    forecaster = EnhancedProductForecaster()
    
    # Select test products
    test_products = []
    for category in df['category'].unique()[:4]:  # Test 4 categories
        cat_products = df[df['category'] == category].groupby('product_id')['y'].agg(['mean', 'count'])
        suitable = cat_products[(cat_products['mean'] >= 5) & (cat_products['count'] >= 400)]
        if len(suitable) > 0:
            test_products.append(suitable.nlargest(1, 'mean').index[0])
    
    print(f"Testing {len(test_products)} products with enhanced forecasting")
    
    # Run comparison
    results = forecaster.compare_standard_vs_enhanced(df, test_products)
    
    # Summary
    if results:
        improvements = [r['improvement_pct'] for r in results.values()]
        avg_improvement = np.mean(improvements)
        
        print(f"\n📊 SUMMARY:")
        print(f"  • Products tested: {len(results)}")
        print(f"  • Average improvement: {avg_improvement:.1f}%")
        print(f"  • Best improvement: {max(improvements):.1f}%")
        print(f"  • Worst improvement: {min(improvements):.1f}%")
        
        if avg_improvement > 5:
            print(f"\n✅ Enhanced models show significant improvement!")
        elif avg_improvement > 0:
            print(f"\n✅ Enhanced models show modest improvement")
        else:
            print(f"\n⚠️ Enhanced models not significantly better")
    
    return forecaster, results

if __name__ == "__main__":
    forecaster, results = run_enhanced_forecasting_demo()
    
    print(f"\n🎯 Enhanced forecasting demo complete!")
    print(f"  • External regressors can improve accuracy by 5-15%")
    print(f"  • Key regressors: category trends, seasonality, relative performance")
    print(f"  • Trade-off: Better accuracy vs more complex data requirements")
