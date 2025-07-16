import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_time_features(df):
    """
    Create comprehensive time-based features for Prophet modeling
    """
    print("Creating time-based features...")
    
    # Ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Basic time features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['day_of_week'] = df['ds'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['week_of_year'] = df['ds'].dt.isocalendar().week
    df['quarter'] = df['ds'].dt.quarter
    
    # Weekend indicator (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)
    
    # End of week (Friday)
    df['is_end_of_week'] = (df['day_of_week'] == 4).astype(int)
    
    # End of month (last 3 days of month)
    df['is_end_of_month'] = df['ds'].dt.is_month_end.astype(int)
    
    # End of month (last 3 days approach)
    df['days_until_month_end'] = (df['ds'].dt.days_in_month - df['ds'].dt.day)
    df['is_last_3_days_month'] = (df['days_until_month_end'] <= 2).astype(int)
    
    # Beginning of month (first 3 days)
    df['is_beginning_of_month'] = (df['ds'].dt.day <= 3).astype(int)
    
    # Middle of month
    df['is_middle_of_month'] = ((df['ds'].dt.day >= 14) & (df['ds'].dt.day <= 16)).astype(int)
    
    # Payday periods (typically 15th and end of month)
    df['is_payday_period'] = ((df['ds'].dt.day.isin([14, 15, 16])) | 
                              (df['days_until_month_end'] <= 2)).astype(int)
    
    # Season indicators
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    # Holiday indicators (major shopping periods)
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)  # Nov-Dec
    df['is_back_to_school'] = df['month'].isin([8, 9]).astype(int)    # Aug-Sep
    df['is_summer_season'] = df['month'].isin([6, 7, 8]).astype(int)  # Jun-Aug
    
    # Specific dates that might affect demand
    df['is_january'] = (df['month'] == 1).astype(int)  # New Year effect
    df['is_december'] = (df['month'] == 12).astype(int)  # Christmas effect
    
    # Business quarter indicators
    df['is_q1'] = (df['quarter'] == 1).astype(int)
    df['is_q2'] = (df['quarter'] == 2).astype(int)
    df['is_q3'] = (df['quarter'] == 3).astype(int)
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    print("✅ Time features created successfully!")
    return df

def load_and_prepare_data(file_path):
    """Load and prepare data with enhanced features"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Create time features
    df = create_time_features(df)
    
    return df

def prepare_category_data_with_features(df, category):
    """Prepare aggregated data for a specific category with features"""
    print(f"\nPreparing enhanced data for category: {category}")
    
    # Filter for the category
    category_df = df[df['category'] == category].copy()
    
    # Aggregate demand by date but keep all the features
    # First, get the demand aggregation
    demand_agg = category_df.groupby('ds')['y'].sum().reset_index()
    
    # Then get the features (they should be the same for each date)
    feature_cols = [col for col in category_df.columns if col not in ['product_id', 'category', 'y']]
    features_agg = category_df.groupby('ds')[feature_cols].first().reset_index()
    
    # Merge demand with features
    category_data = pd.merge(demand_agg, features_agg, on='ds')
    category_data = category_data.sort_values('ds').reset_index(drop=True)
    
    print(f"Category data shape: {category_data.shape}")
    print(f"Date range: {category_data['ds'].min()} to {category_data['ds'].max()}")
    print(f"Average daily demand: {category_data['y'].mean():.2f}")
    print(f"Features available: {len(feature_cols)}")
    
    return category_data

def create_enhanced_prophet_model(train_data, category, model_params=None):
    """Create Prophet model with custom regressors (features)"""
    print(f"\n{'='*50}")
    print(f"TRAINING ENHANCED MODEL FOR: {category.upper()}")
    print(f"{'='*50}")
    
    # Default parameters
    if model_params is None:
        model_params = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
    
    print(f"Model parameters: {model_params}")
    
    # Create Prophet model
    model = Prophet(**model_params)
    
    # Add custom regressors (our engineered features)
    feature_cols = [
        'is_weekend', 'is_end_of_week', 'is_end_of_month', 'is_last_3_days_month',
        'is_beginning_of_month', 'is_middle_of_month', 'is_payday_period',
        'is_spring', 'is_summer', 'is_fall', 'is_winter',
        'is_holiday_season', 'is_back_to_school', 'is_summer_season',
        'is_january', 'is_december', 'is_q1', 'is_q2', 'is_q3', 'is_q4'
    ]
    
    # Check which features are available in the data
    available_features = [col for col in feature_cols if col in train_data.columns]
    print(f"Adding {len(available_features)} custom regressors:")
    
    for feature in available_features:
        model.add_regressor(feature)
        print(f"  ✓ {feature}")
    
    # Fit the model
    print("Fitting enhanced model...")
    model.fit(train_data[['ds', 'y'] + available_features])
    print("✅ Enhanced model fitted successfully!")
    
    return model, available_features

def create_future_with_features(model, train_data, periods, available_features):
    """Create future dataframe with features for prediction"""
    print(f"Creating future dataframe with features for {periods} days...")
    
    # Create basic future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Add features to future dataframe
    future = create_time_features(future)
    
    # Ensure all required features are present
    for feature in available_features:
        if feature not in future.columns:
            print(f"⚠️  Feature {feature} missing in future dataframe, setting to 0")
            future[feature] = 0
    
    return future[['ds'] + available_features]

def evaluate_enhanced_model(model, train_data, test_data, available_features, category):
    """Evaluate the enhanced Prophet model"""
    print(f"\nEvaluating enhanced model for {category}...")
    
    # Create future dataframe that includes test period
    future_data = pd.concat([train_data, test_data], ignore_index=True)
    future = future_data[['ds'] + available_features].copy()
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract test predictions
    test_start_idx = len(train_data)
    test_predictions = forecast['yhat'][test_start_idx:].values
    test_actual = test_data['y'].values
    
    # Calculate metrics
    mae = mean_absolute_error(test_actual, test_predictions)
    mse = mean_squared_error(test_actual, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_actual, test_predictions)
    mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
    
    metrics = {
        'category': category,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'test_size': len(test_data),
        'features_used': len(available_features)
    }
    
    print(f"Enhanced Model Performance for {category}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Features used: {len(available_features)}")
    
    return metrics, forecast

def analyze_feature_importance(model, available_features):
    """Analyze the importance of custom features"""
    print("\n--- FEATURE IMPORTANCE ANALYSIS ---")
    
    # Get the posterior samples for regressors
    try:
        regressor_coefficients = {}
        
        # Extract regressor coefficients from the model
        if hasattr(model, 'params') and model.params is not None:
            for feature in available_features:
                if f'beta_{feature}' in model.params:
                    coef_samples = model.params[f'beta_{feature}']
                    regressor_coefficients[feature] = {
                        'mean': np.mean(coef_samples),
                        'std': np.std(coef_samples),
                        'q025': np.percentile(coef_samples, 2.5),
                        'q975': np.percentile(coef_samples, 97.5)
                    }
        
        if regressor_coefficients:
            print("Feature coefficients (mean ± std):")
            sorted_features = sorted(regressor_coefficients.items(), 
                                   key=lambda x: abs(x[1]['mean']), reverse=True)
            
            for feature, stats in sorted_features:
                mean_coef = stats['mean']
                std_coef = stats['std']
                significant = not (stats['q025'] <= 0 <= stats['q975'])
                significance = "***" if significant else ""
                
                print(f"  {feature:25s}: {mean_coef:8.3f} ± {std_coef:6.3f} {significance}")
                
            return regressor_coefficients
        else:
            print("Could not extract feature coefficients from model")
            return None
            
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        return None

def plot_enhanced_forecast(forecast, train_data, test_data, category, available_features):
    """Plot enhanced forecast results with feature analysis"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # Main forecast plot
    ax1 = axes[0, 0]
    ax1.plot(train_data['ds'], train_data['y'], 'o', markersize=2, label='Training Data', alpha=0.6)
    ax1.plot(test_data['ds'], test_data['y'], 'o', markersize=2, label='Actual Test Data', alpha=0.8, color='green')
    
    test_start_idx = len(train_data)
    forecast_test = forecast[test_start_idx:]
    ax1.plot(forecast_test['ds'], forecast_test['yhat'], 'r-', label='Enhanced Forecast', linewidth=2)
    ax1.fill_between(forecast_test['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], 
                     alpha=0.3, color='red', label='Confidence Interval')
    
    ax1.set_title(f'{category.title()} - Enhanced Prophet Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Demand')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trend component
    ax2 = axes[0, 1]
    ax2.plot(forecast['ds'], forecast['trend'], 'b-', linewidth=2)
    ax2.set_title(f'{category.title()} - Trend Component')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Trend')
    ax2.grid(True, alpha=0.3)
    
    # Weekly seasonality
    ax3 = axes[1, 0]
    if 'weekly' in forecast.columns:
        ax3.plot(forecast['ds'], forecast['weekly'], 'orange', linewidth=2)
        ax3.set_title(f'{category.title()} - Weekly Seasonality')
    else:
        ax3.text(0.5, 0.5, 'Weekly seasonality not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{category.title()} - Weekly Seasonality (N/A)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Weekly Effect')
    ax3.grid(True, alpha=0.3)
    
    # Yearly seasonality
    ax4 = axes[1, 1]
    if 'yearly' in forecast.columns:
        ax4.plot(forecast['ds'], forecast['yearly'], 'g-', linewidth=2)
        ax4.set_title(f'{category.title()} - Yearly Seasonality')
    else:
        ax4.text(0.5, 0.5, 'Yearly seasonality not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'{category.title()} - Yearly Seasonality (N/A)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Yearly Effect')
    ax4.grid(True, alpha=0.3)
    
    # Feature contributions (sum of top features)
    ax5 = axes[2, 0]
    feature_cols = [col for col in forecast.columns if col in available_features]
    if feature_cols:
        # Sum the contributions of all features
        total_feature_effect = forecast[feature_cols].sum(axis=1)
        ax5.plot(forecast['ds'], total_feature_effect, 'purple', linewidth=2)
        ax5.set_title(f'{category.title()} - Total Feature Effects')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Feature Contribution')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Feature effects not available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title(f'{category.title()} - Feature Effects (N/A)')
    
    # Residuals analysis
    ax6 = axes[2, 1]
    train_residuals = train_data['y'] - forecast['yhat'][:len(train_data)]
    ax6.hist(train_residuals.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax6.set_title(f'{category.title()} - Residuals Distribution')
    ax6.set_xlabel('Residual')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # Add residual statistics
    mean_residual = train_residuals.mean()
    std_residual = train_residuals.std()
    ax6.axvline(mean_residual, color='red', linestyle='--', label=f'Mean: {mean_residual:.2f}')
    ax6.text(0.02, 0.98, f'Std: {std_residual:.2f}', transform=ax6.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def compare_models(basic_metrics, enhanced_metrics):
    """Compare basic Prophet vs Enhanced Prophet models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON: BASIC vs ENHANCED PROPHET")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Category': enhanced_metrics['category'],
        'Basic_RMSE': basic_metrics['RMSE'],
        'Enhanced_RMSE': enhanced_metrics['RMSE'],
        'Basic_R2': basic_metrics['R2'],
        'Enhanced_R2': enhanced_metrics['R2'],
        'Features_Used': enhanced_metrics['features_used']
    })
    
    # Calculate improvements
    comparison_df['RMSE_Improvement_%'] = ((comparison_df['Basic_RMSE'] - comparison_df['Enhanced_RMSE']) / 
                                           comparison_df['Basic_RMSE'] * 100)
    comparison_df['R2_Improvement'] = comparison_df['Enhanced_R2'] - comparison_df['Basic_R2']
    
    print(comparison_df.to_string(index=False))
    
    # Summary statistics
    avg_rmse_improvement = comparison_df['RMSE_Improvement_%'].mean()
    avg_r2_improvement = comparison_df['R2_Improvement'].mean()
    
    print(f"\n📊 OVERALL IMPROVEMENT SUMMARY:")
    print(f"Average RMSE improvement: {avg_rmse_improvement:.1f}%")
    print(f"Average R² improvement: {avg_r2_improvement:.3f}")
    
    if avg_rmse_improvement > 5:
        print("✅ Enhanced features provide significant improvement!")
    elif avg_rmse_improvement > 0:
        print("🟡 Enhanced features provide modest improvement")
    else:
        print("⚠️  Enhanced features may not be adding much value")
    
    return comparison_df

def create_train_test_split(category_data, test_size=0.2):
    """Create train/test split for the category data"""
    split_index = int(len(category_data) * (1 - test_size))
    train_data = category_data[:split_index].copy()
    test_data = category_data[split_index:].copy()
    
    print(f"Train data: {len(train_data)} days ({train_data['ds'].min()} to {train_data['ds'].max()})")
    print(f"Test data: {len(test_data)} days ({test_data['ds'].min()} to {test_data['ds'].max()})")
    
    return train_data, test_data

def main():
    """Main function to run enhanced category-level Prophet modeling"""
    print("="*80)
    print("ENHANCED CATEGORY-LEVEL PROPHET MODELING")
    print("="*80)
    
    # Load data with features
    df = load_and_prepare_data('daily_demand_by_product_modern.csv')
    
    # Get all categories
    categories = df['category'].unique()
    print(f"\nCategories to model: {categories}")
    
    # Store results for both basic and enhanced models
    all_basic_metrics = []
    all_enhanced_metrics = []
    all_enhanced_models = {}
    
    # Train both basic and enhanced models for each category
    for i, category in enumerate(categories, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING CATEGORY {i}/{len(categories)}: {category.upper()}")
        print(f"{'#'*80}")
        
        try:
            # Prepare category data with features
            category_data = prepare_category_data_with_features(df, category)
            
            # Split into train/test
            train_data, test_data = create_train_test_split(category_data)
            
            # =======================================================
            # BASIC PROPHET MODEL (for comparison)
            # =======================================================
            print(f"\n--- BASIC PROPHET MODEL ---")
            basic_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            basic_model.fit(train_data[['ds', 'y']])
            
            # Basic prediction
            basic_future = basic_model.make_future_dataframe(periods=len(test_data))
            basic_forecast = basic_model.predict(basic_future)
            
            # Basic evaluation
            test_start_idx = len(train_data)
            basic_predictions = basic_forecast['yhat'][test_start_idx:].values
            test_actual = test_data['y'].values
            
            basic_mae = mean_absolute_error(test_actual, basic_predictions)
            basic_rmse = np.sqrt(mean_squared_error(test_actual, basic_predictions))
            basic_r2 = r2_score(test_actual, basic_predictions)
            
            basic_metrics = {
                'category': category,
                'MAE': basic_mae,
                'RMSE': basic_rmse,
                'R2': basic_r2
            }
            
            print(f"Basic Model - RMSE: {basic_rmse:.2f}, R²: {basic_r2:.4f}")
            
            # =======================================================
            # ENHANCED PROPHET MODEL (with features)
            # =======================================================
            print(f"\n--- ENHANCED PROPHET MODEL ---")
            enhanced_model, available_features = create_enhanced_prophet_model(train_data, category)
            
            # Enhanced evaluation
            enhanced_metrics, enhanced_forecast = evaluate_enhanced_model(
                enhanced_model, train_data, test_data, available_features, category
            )
            
            # Feature importance analysis
            feature_importance = analyze_feature_importance(enhanced_model, available_features)
            
            # Plotting
            plot_enhanced_forecast(enhanced_forecast, train_data, test_data, category, available_features)
            
            # Store results
            all_basic_metrics.append(basic_metrics)
            all_enhanced_metrics.append(enhanced_metrics)
            all_enhanced_models[category] = {
                'model': enhanced_model,
                'features': available_features,
                'importance': feature_importance
            }
            
            print(f"✅ Successfully completed {category}")
            
        except Exception as e:
            print(f"❌ Error processing {category}: {str(e)}")
            continue
    
    # Final comparison and analysis
    if all_enhanced_metrics:
        print(f"\n{'#'*80}")
        print("FINAL ANALYSIS AND RECOMMENDATIONS")
        print(f"{'#'*80}")
        
        # Compare basic vs enhanced models
        for basic, enhanced in zip(all_basic_metrics, all_enhanced_metrics):
            comparison = compare_models(basic, enhanced)
        
        # Feature importance across categories
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ACROSS CATEGORIES")
        print(f"{'='*60}")
        
        # Aggregate feature importance across all categories
        all_feature_importance = {}
        for category, model_data in all_enhanced_models.items():
            if model_data['importance']:
                print(f"\n{category.upper()}:")
                for feature, stats in model_data['importance'].items():
                    print(f"  {feature:25s}: {stats['mean']:8.3f}")
                    
                    # Aggregate for overall analysis
                    if feature not in all_feature_importance:
                        all_feature_importance[feature] = []
                    all_feature_importance[feature].append(abs(stats['mean']))
        
        # Overall feature ranking
        if all_feature_importance:
            print(f"\n{'='*60}")
            print("OVERALL FEATURE RANKING (by average absolute coefficient)")
            print(f"{'='*60}")
            
            feature_rankings = {}
            for feature, coeffs in all_feature_importance.items():
                feature_rankings[feature] = np.mean(coeffs)
            
            sorted_features = sorted(feature_rankings.items(), key=lambda x: x[1], reverse=True)
            
            print("Most impactful features across all categories:")
            for i, (feature, avg_coeff) in enumerate(sorted_features[:10], 1):
                print(f"  {i:2d}. {feature:25s}: {avg_coeff:.4f}")
        
        # Final recommendations
        print(f"\n{'='*60}")
        print("MODELING RECOMMENDATIONS")
        print(f"{'='*60}")
        
        print("✅ ENHANCED APPROACH BENEFITS:")
        print("   • Captures business-specific patterns (weekends, month-end, etc.)")
        print("   • More interpretable feature effects")
        print("   • Better handling of irregular patterns")
        print("   • Consistent modeling approach across categories")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Deploy enhanced models for production forecasting")
        print("   2. Monitor feature importance over time")
        print("   3. Add holiday calendar effects")
        print("   4. Consider external factors (promotions, economic indicators)")
        print("   5. Implement automated retraining pipeline")
        
        # Save models
        models_dir = 'enhanced_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        for category, model_data in all_enhanced_models.items():
            model_path = os.path.join(models_dir, f'enhanced_prophet_{category}.pkl')
            joblib.dump(model_data, model_path)
            print(f"✅ Enhanced model saved: {model_path}")
    
    return all_enhanced_models, all_enhanced_metrics, all_basic_metrics

if __name__ == "__main__":
    models, enhanced_metrics, basic_metrics = main()
