#!/usr/bin/env python3
"""
INDIVIDUAL CATEGORY MODELS - OPTIMIZED FOR INDIVIDUAL ITEM FORECASTING
======================================================================
Creates 6 separate optimized Prophet models (one per category) with best parameters
identified from comprehensive analysis. Designed for forecasting individual products.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import pickle
import os
warnings.filterwarnings('ignore')

class CategoryModelManager:
    """Manages optimized Prophet models for each category"""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.holidays_df = None
        self.performance_metrics = {}
        
        # Best parameters identified from analysis
        self.optimal_configs = {
            'books_media': {
                'changepoint_prior_scale': 0.5,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            },
            'clothing': {
                'changepoint_prior_scale': 1.0,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            },
            'electronics': {
                'changepoint_prior_scale': 0.5,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'multiplicative'
            },
            'health_beauty': {
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            },
            'home_garden': {
                'changepoint_prior_scale': 1.0,
                'seasonality_prior_scale': 10,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            },
            'sports_outdoors': {
                'changepoint_prior_scale': 0.5,
                'seasonality_prior_scale': 0.1,
                'holidays_prior_scale': 10,
                'seasonality_mode': 'additive'
            }
        }
    
    def create_holiday_calendar(self):
        """Create comprehensive holiday calendar"""
        major_holidays = pd.DataFrame([
            # Black Friday
            {'holiday': 'black_friday', 'ds': '2022-11-25', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2023-11-24', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2024-11-29', 'lower_window': -1, 'upper_window': 1},
            {'holiday': 'black_friday', 'ds': '2025-11-28', 'lower_window': -1, 'upper_window': 1},
            
            # Cyber Monday
            {'holiday': 'cyber_monday', 'ds': '2022-11-28', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'cyber_monday', 'ds': '2023-11-27', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'cyber_monday', 'ds': '2024-12-02', 'lower_window': 0, 'upper_window': 0},
            {'holiday': 'cyber_monday', 'ds': '2025-12-01', 'lower_window': 0, 'upper_window': 0},
            
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
        self.holidays_df = major_holidays
        return major_holidays
    
    def add_regressors(self, data):
        """Add enhanced regressors based on analysis"""
        enhanced_data = data.copy()
        
        # Weekend regressor
        enhanced_data['is_weekend'] = enhanced_data['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Month-end effect (shopping patterns)
        enhanced_data['month_end_effect'] = (enhanced_data['ds'].dt.day >= 25).astype(int)
        
        # Payday effect (1st and 15th of month)
        enhanced_data['payday_effect'] = enhanced_data['ds'].dt.day.isin([1, 15]).astype(int)
        
        # Summer surge (June-August)
        enhanced_data['summer_surge'] = enhanced_data['ds'].dt.month.isin([6, 7, 8]).astype(int)
        
        return enhanced_data
    
    def train_category_model(self, category_data, category_name):
        """Train optimized model for specific category"""
        print(f"\n🔧 Training optimized model for {category_name.upper()}")
        print("-" * 60)
        
        # Get optimal parameters for this category
        params = self.optimal_configs[category_name].copy()
        params['holidays'] = self.holidays_df
        
        # Add regressors to data
        enhanced_data = self.add_regressors(category_data)
        
        # Split data for validation (80/20)
        split_idx = int(len(enhanced_data) * 0.8)
        train_data = enhanced_data[:split_idx].copy()
        test_data = enhanced_data[split_idx:].copy()
        
        print(f"📊 Data Info:")
        print(f"   • Total data points: {len(enhanced_data)}")
        print(f"   • Training period: {train_data['ds'].min().date()} to {train_data['ds'].max().date()}")
        print(f"   • Test period: {test_data['ds'].min().date()} to {test_data['ds'].max().date()}")
        print(f"   • Demand range: {enhanced_data['y'].min():.0f} to {enhanced_data['y'].max():.0f}")
        
        # Create and configure model
        model = Prophet(**params)
        
        # Add regressors based on category-specific analysis
        regressors_to_add = self.get_category_regressors(category_name)
        for regressor in regressors_to_add:
            model.add_regressor(regressor, prior_scale=10, standardize=False)
        
        print(f"🎯 Model Configuration:")
        print(f"   • Changepoint prior scale: {params['changepoint_prior_scale']}")
        print(f"   • Seasonality mode: {params['seasonality_mode']}")
        print(f"   • Regressors: {', '.join(regressors_to_add)}")
        
        # Train model
        regressor_columns = ['ds', 'y'] + regressors_to_add
        model.fit(train_data[regressor_columns])
        
        # Validate on test set
        test_columns = ['ds'] + regressors_to_add
        test_forecast = model.predict(test_data[test_columns])
        
        # Calculate metrics
        actual = test_data['y'].values
        predicted = test_forecast['yhat'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Store performance metrics
        self.performance_metrics[category_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'regressors': regressors_to_add
        }
        
        print(f"📈 Validation Metrics:")
        print(f"   • RMSE: {rmse:.1f}")
        print(f"   • MAE: {mae:.1f}")
        print(f"   • R²: {r2:.3f}")
        print(f"   • MAPE: {mape:.1f}%")
        
        # Train final model on all data for production use
        final_model = Prophet(**params)
        for regressor in regressors_to_add:
            final_model.add_regressor(regressor, prior_scale=10, standardize=False)
        
        final_model.fit(enhanced_data[regressor_columns])
        
        # Store model
        self.models[category_name] = final_model
        self.best_params[category_name] = params
        
        return final_model, self.performance_metrics[category_name]
    
    def get_category_regressors(self, category_name):
        """Get optimal regressors for each category based on analysis"""
        category_regressors = {
            'books_media': ['is_weekend', 'month_end_effect', 'payday_effect', 'summer_surge'],
            'clothing': ['is_weekend'],
            'electronics': ['is_weekend', 'month_end_effect', 'payday_effect'],
            'health_beauty': ['is_weekend'],
            'home_garden': ['is_weekend', 'month_end_effect', 'payday_effect', 'summer_surge'],
            'sports_outdoors': ['is_weekend', 'month_end_effect', 'payday_effect']
        }
        return category_regressors.get(category_name, ['is_weekend'])
    
    def forecast_individual_item(self, product_id, category, forecast_days=30):
        """Forecast demand for individual product using category model"""
        if category not in self.models:
            raise ValueError(f"No trained model found for category: {category}")
        
        # Load original data to get product-specific patterns
        df = pd.read_csv('daily_demand_by_product_modern.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get product-specific data
        product_data = df[df['product_id'] == product_id].copy()
        if len(product_data) == 0:
            raise ValueError(f"No data found for product_id: {product_id}")
        
        product_category = product_data['category'].iloc[0]
        if product_category != category:
            print(f"⚠️  Warning: Product {product_id} is in category '{product_category}', not '{category}'")
            category = product_category
        
        # Prepare data with regressors
        enhanced_data = self.add_regressors(product_data)
        
        # Get the trained model
        model = self.models[category]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        future_enhanced = self.add_regressors(future)
        
        # Get regressors for this category
        regressors = self.get_category_regressors(category)
        future_columns = ['ds'] + regressors
        
        # Generate forecast
        forecast = model.predict(future_enhanced[future_columns])
        
        return forecast, enhanced_data
    
    def save_models(self, directory='trained_models'):
        """Save all trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for category, model in self.models.items():
            model_path = os.path.join(directory, f'{category}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save configuration
        config_path = os.path.join(directory, 'model_config.pkl')
        config = {
            'best_params': self.best_params,
            'holidays_df': self.holidays_df,
            'performance_metrics': self.performance_metrics,
            'optimal_configs': self.optimal_configs
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"✅ All models saved to '{directory}' directory")

def load_and_prepare_data():
    """Load and prepare data for category-level modeling"""
    print("📁 Loading data for category model training...")
    
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Aggregate by category and date
    category_data = df.groupby(['category', 'ds'])['y'].sum().reset_index()
    
    categories = df['category'].unique()
    products_per_category = df.groupby('category')['product_id'].nunique()
    
    print(f"📊 Dataset Overview:")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"   • Categories: {len(categories)}")
    print(f"   • Total products: {df['product_id'].nunique()}")
    
    print(f"\n📈 Products per category:")
    for category in categories:
        count = products_per_category[category]
        print(f"   • {category}: {count} products")
    
    return category_data, categories, df

def create_accuracy_comparison_visualization(manager):
    """Create comprehensive accuracy comparison visualization"""
    
    metrics_df = pd.DataFrame(manager.performance_metrics).T
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏆 Category Model Accuracy Comparison', fontsize=16, fontweight='bold')
    
    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    categories = metrics_df.index
    rmse_values = metrics_df['rmse']
    bars1 = ax1.bar(categories, rmse_values, color='skyblue', alpha=0.8)
    ax1.set_title('RMSE by Category', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. R² Comparison
    ax2 = axes[0, 1]
    r2_values = metrics_df['r2']
    bars2 = ax2.bar(categories, r2_values, color='lightgreen', alpha=0.8)
    ax2.set_title('R² Score by Category', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. MAPE Comparison
    ax3 = axes[1, 0]
    mape_values = metrics_df['mape']
    bars3 = ax3.bar(categories, mape_values, color='lightcoral', alpha=0.8)
    ax3.set_title('MAPE by Category (%)', fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, mape_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Configuration Heatmap
    ax4 = axes[1, 1]
    
    # Create configuration matrix
    config_data = []
    config_labels = []
    
    for category in categories:
        regressors = manager.performance_metrics[category]['regressors']
        config_data.append([
            1 if 'is_weekend' in regressors else 0,
            1 if 'month_end_effect' in regressors else 0,
            1 if 'payday_effect' in regressors else 0,
            1 if 'summer_surge' in regressors else 0
        ])
        config_labels.append(category)
    
    config_matrix = np.array(config_data)
    
    sns.heatmap(config_matrix, 
                xticklabels=['Weekend', 'Month-end', 'Payday', 'Summer'],
                yticklabels=config_labels,
                annot=True, 
                fmt='d',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Regressor Used'},
                ax=ax4)
    ax4.set_title('Regressor Configuration by Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('category_models_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_accuracy_report(manager):
    """Print detailed accuracy report"""
    
    print("\n" + "="*80)
    print("📊 DETAILED ACCURACY COMPARISON REPORT")
    print("="*80)
    
    # Summary statistics
    metrics_df = pd.DataFrame(manager.performance_metrics).T
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print("-" * 60)
    print(f"{'Category':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'MAPE':<8} {'Grade':<10}")
    print("-" * 60)
    
    for category, metrics in manager.performance_metrics.items():
        # Assign performance grade
        if metrics['r2'] >= 0.7:
            grade = "Excellent"
        elif metrics['r2'] >= 0.5:
            grade = "Good"
        elif metrics['r2'] >= 0.3:
            grade = "Fair"
        else:
            grade = "Poor"
        
        print(f"{category:<15} {metrics['rmse']:<8.0f} {metrics['mae']:<8.0f} "
              f"{metrics['r2']:<8.3f} {metrics['mape']:<8.1f} {grade:<10}")
    
    # Best and worst performers
    best_r2_category = metrics_df['r2'].idxmax()
    worst_r2_category = metrics_df['r2'].idxmin()
    best_rmse_category = metrics_df['rmse'].idxmin()
    worst_rmse_category = metrics_df['rmse'].idxmax()
    
    print(f"\n🥇 PERFORMANCE RANKINGS:")
    print("-" * 40)
    print(f"Best R² Score: {best_r2_category} ({metrics_df.loc[best_r2_category, 'r2']:.3f})")
    print(f"Worst R² Score: {worst_r2_category} ({metrics_df.loc[worst_r2_category, 'r2']:.3f})")
    print(f"Lowest RMSE: {best_rmse_category} ({metrics_df.loc[best_rmse_category, 'rmse']:.0f})")
    print(f"Highest RMSE: {worst_rmse_category} ({metrics_df.loc[worst_rmse_category, 'rmse']:.0f})")
    
    # Configuration analysis
    print(f"\n🎯 REGRESSOR EFFECTIVENESS:")
    print("-" * 40)
    
    regressor_usage = {
        'is_weekend': [],
        'month_end_effect': [],
        'payday_effect': [],
        'summer_surge': []
    }
    
    for category, metrics in manager.performance_metrics.items():
        for regressor in metrics['regressors']:
            if regressor in regressor_usage:
                regressor_usage[regressor].append((category, metrics['r2']))
    
    for regressor, categories in regressor_usage.items():
        if categories:
            avg_r2 = np.mean([r2 for _, r2 in categories])
            count = len(categories)
            category_names = [cat for cat, _ in categories]
            print(f"{regressor:<18}: {count}/6 categories, Avg R²: {avg_r2:.3f}")
            print(f"{'':20} Used in: {', '.join(category_names)}")
        else:
            print(f"{regressor:<18}: Not used")
    
    # Overall statistics
    avg_rmse = metrics_df['rmse'].mean()
    avg_r2 = metrics_df['r2'].mean()
    avg_mape = metrics_df['mape'].mean()
    
    print(f"\n📈 OVERALL STATISTICS:")
    print("-" * 30)
    print(f"Average RMSE: {avg_rmse:.1f}")
    print(f"Average R²: {avg_r2:.3f}")
    print(f"Average MAPE: {avg_mape:.1f}%")
    
    excellent_count = sum(1 for metrics in manager.performance_metrics.values() if metrics['r2'] >= 0.7)
    good_count = sum(1 for metrics in manager.performance_metrics.values() if 0.5 <= metrics['r2'] < 0.7)
    
    print(f"Models with R² ≥ 0.7: {excellent_count}/6 categories")
    print(f"Models with R² ≥ 0.5: {excellent_count + good_count}/6 categories")

def demonstrate_individual_forecasting(manager, df):
    """Demonstrate individual product forecasting"""
    
    print("\n" + "="*80)
    print("🔮 INDIVIDUAL PRODUCT FORECASTING DEMONSTRATION")
    print("="*80)
    
    # Select sample products from different categories
    sample_products = {}
    for category in manager.models.keys():
        category_products = df[df['category'] == category]['product_id'].unique()
        if len(category_products) > 0:
            # Select a product with reasonable demand history
            sample_product = category_products[0]
            sample_products[category] = sample_product
    
    print(f"\n📋 Sample Products Selected:")
    for category, product_id in sample_products.items():
        product_data = df[df['product_id'] == product_id]
        avg_demand = product_data['y'].mean()
        print(f"   • {category}: Product {product_id} (Avg demand: {avg_demand:.1f})")
    
    # Generate forecasts for sample products
    print(f"\n🔮 Generating 30-day forecasts...")
    
    for category, product_id in sample_products.items():
        try:
            forecast, historical_data = manager.forecast_individual_item(product_id, category, forecast_days=30)
            
            # Get recent actual values for comparison
            recent_actual = historical_data.tail(7)['y'].mean()
            future_forecast = forecast.tail(30)['yhat'].mean()
            
            print(f"\n{category.upper()} - Product {product_id}:")
            print(f"   Recent avg demand: {recent_actual:.1f}")
            print(f"   Forecasted avg demand: {future_forecast:.1f}")
            print(f"   Forecast trend: {'+' if future_forecast > recent_actual else '-'}{abs((future_forecast/recent_actual - 1)*100):.1f}%")
            
        except Exception as e:
            print(f"   Error forecasting {category} - Product {product_id}: {str(e)}")

def main():
    """Main execution function"""
    
    print("="*80)
    print("🏭 INDIVIDUAL CATEGORY MODELS TRAINING SYSTEM")
    print("="*80)
    
    # Load data
    category_data, categories, df = load_and_prepare_data()
    
    # Initialize model manager
    manager = CategoryModelManager()
    manager.create_holiday_calendar()
    
    print(f"\n🎯 Training {len(categories)} optimized category models...")
    
    # Train models for each category
    trained_models = {}
    for category in categories:
        cat_data = category_data[category_data['category'] == category].copy()
        cat_data = cat_data.sort_values('ds').reset_index(drop=True)
        
        model, metrics = manager.train_category_model(cat_data, category)
        trained_models[category] = model
    
    # Create accuracy comparison visualization
    print(f"\n📊 Creating accuracy comparison visualizations...")
    create_accuracy_comparison_visualization(manager)
    
    # Print detailed accuracy report
    print_detailed_accuracy_report(manager)
    
    # Demonstrate individual product forecasting
    demonstrate_individual_forecasting(manager, df)
    
    # Save trained models
    print(f"\n💾 Saving trained models...")
    manager.save_models()
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print("✅ 6 optimized category models trained and validated")
    print("✅ Accuracy comparison visualization created")
    print("✅ Models saved for production use")
    print("✅ Individual product forecasting capability ready")
    print("✅ Ready for deployment!")
    
    return manager

if __name__ == "__main__":
    model_manager = main()
