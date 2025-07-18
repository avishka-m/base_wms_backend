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

def load_and_prepare_data(file_path):
    """Load and prepare data for category-level Prophet modeling"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Categories: {df['category'].unique()}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    
    # Convert date column to datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    return df

def prepare_category_data(df, category):
    """Prepare aggregated data for a specific category"""
    print(f"\nPreparing data for category: {category}")
    
    # Filter for the category and aggregate by date
    category_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
    category_data = category_data.sort_values('ds').reset_index(drop=True)
    
    print(f"Category data shape: {category_data.shape}")
    print(f"Date range: {category_data['ds'].min()} to {category_data['ds'].max()}")
    print(f"Average daily demand: {category_data['y'].mean():.2f}")
    print(f"Total demand: {category_data['y'].sum():,.0f}")
    
    return category_data

def create_train_test_split(category_data, test_size=0.2):
    """Create train/test split for the category data"""
    split_index = int(len(category_data) * (1 - test_size))
    train_data = category_data[:split_index].copy()
    test_data = category_data[split_index:].copy()
    
    print(f"Train data: {len(train_data)} days ({train_data['ds'].min()} to {train_data['ds'].max()})")
    print(f"Test data: {len(test_data)} days ({test_data['ds'].min()} to {test_data['ds'].max()})")
    
    return train_data, test_data

def train_category_model(train_data, category, model_params=None):
    """Train Prophet model for a specific category"""
    print(f"\n{'='*50}")
    print(f"TRAINING MODEL FOR: {category.upper()}")
    print(f"{'='*50}")
    
    # Default parameters (can be customized per category)
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
    
    # Create and fit the model
    model = Prophet(**model_params)
    model.fit(train_data)
    
    print(f"✅ Model trained successfully for {category}")
    
    return model

def evaluate_model(model, train_data, test_data, category):
    """Evaluate the trained model"""
    print(f"\nEvaluating model for {category}...")
    
    # Create future dataframe for the test period
    future = model.make_future_dataframe(periods=len(test_data))
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
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100
    
    metrics = {
        'category': category,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'test_size': len(test_data)
    }
    
    print(f"Model Performance for {category}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return metrics, forecast

def save_model(model, category, models_dir='models'):
    """Save the trained model"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, f'prophet_model_{category}.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    return model_path

def plot_category_forecast(forecast, train_data, test_data, category):
    """Plot forecast results for a category"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main forecast plot
    ax1 = axes[0, 0]
    
    # Plot training data
    ax1.plot(train_data['ds'], train_data['y'], 'o', markersize=2, label='Training Data', alpha=0.6)
    
    # Plot test data
    ax1.plot(test_data['ds'], test_data['y'], 'o', markersize=2, label='Actual Test Data', alpha=0.8, color='green')
    
    # Plot forecast
    test_start_idx = len(train_data)
    forecast_test = forecast[test_start_idx:]
    ax1.plot(forecast_test['ds'], forecast_test['yhat'], 'r-', label='Forecast', linewidth=2)
    
    # Plot confidence intervals for test period
    ax1.fill_between(forecast_test['ds'], 
                     forecast_test['yhat_lower'], 
                     forecast_test['yhat_upper'], 
                     alpha=0.3, color='red', label='Confidence Interval')
    
    ax1.set_title(f'{category.title()} - Prophet Forecast')
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
    
    # Yearly seasonality (if available)
    ax3 = axes[1, 0]
    if 'yearly' in forecast.columns:
        ax3.plot(forecast['ds'], forecast['yearly'], 'g-', linewidth=2)
        ax3.set_title(f'{category.title()} - Yearly Seasonality')
    else:
        ax3.text(0.5, 0.5, 'Yearly seasonality not available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{category.title()} - Yearly Seasonality (N/A)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Yearly Effect')
    ax3.grid(True, alpha=0.3)
    
    # Weekly seasonality (if available)
    ax4 = axes[1, 1]
    if 'weekly' in forecast.columns:
        ax4.plot(forecast['ds'], forecast['weekly'], 'orange', linewidth=2)
        ax4.set_title(f'{category.title()} - Weekly Seasonality')
    else:
        ax4.text(0.5, 0.5, 'Weekly seasonality not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'{category.title()} - Weekly Seasonality (N/A)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Weekly Effect')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_category_performance(all_metrics):
    """Compare performance across all categories"""
    print("\n" + "="*60)
    print("CATEGORY MODEL COMPARISON")
    print("="*60)
    
    # Create DataFrame for easy comparison
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.sort_values('RMSE')
    
    print("\nRanking by RMSE (lower is better):")
    print(df_metrics[['category', 'RMSE', 'MAE', 'R2', 'MAPE']].to_string(index=False))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    categories = df_metrics['category']
    
    # RMSE comparison
    axes[0, 0].bar(categories, df_metrics['RMSE'])
    axes[0, 0].set_title('RMSE by Category')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[0, 1].bar(categories, df_metrics['MAE'])
    axes[0, 1].set_title('MAE by Category')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R² comparison
    axes[1, 0].bar(categories, df_metrics['R2'])
    axes[1, 0].set_title('R² Score by Category')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # MAPE comparison
    axes[1, 1].bar(categories, df_metrics['MAPE'])
    axes[1, 1].set_title('MAPE by Category (%)')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df_metrics

def create_interactive_comparison(all_metrics, all_forecasts, all_test_data):
    """Create interactive plots comparing all categories"""
    # Create subplots for all categories
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"{cat.title()}" for cat in all_metrics['category']],
        vertical_spacing=0.08
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (category, forecast, test_data) in enumerate(zip(
        all_metrics['category'], all_forecasts, all_test_data
    )):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Add actual data
        fig.add_trace(
            go.Scatter(
                x=test_data['ds'],
                y=test_data['y'],
                mode='markers',
                name=f'{category} - Actual',
                marker=dict(color=colors[i], size=3),
                showlegend=True if i == 0 else False
            ),
            row=row, col=col
        )
        
        # Add forecast
        test_start_idx = len(forecast) - len(test_data)
        forecast_test = forecast[test_start_idx:]
        
        fig.add_trace(
            go.Scatter(
                x=forecast_test['ds'],
                y=forecast_test['yhat'],
                mode='lines',
                name=f'{category} - Forecast',
                line=dict(color=colors[i], dash='dash'),
                showlegend=True if i == 0 else False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Category-wise Prophet Forecasts Comparison',
        height=900,
        showlegend=True
    )
    
    fig.show()

def main():
    """Main function to train category-level Prophet models"""
    print("="*80)
    print("CATEGORY-LEVEL PROPHET MODELING")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data('daily_demand_by_product_modern.csv')
    
    # Get all categories
    categories = df['category'].unique()
    print(f"\nCategories to model: {categories}")
    
    # Store results
    all_models = {}
    all_metrics = []
    all_forecasts = []
    all_test_data = []
    
    # Train model for each category
    for i, category in enumerate(categories, 1):
        print(f"\n{'#'*80}")
        print(f"PROCESSING CATEGORY {i}/{len(categories)}: {category.upper()}")
        print(f"{'#'*80}")
        
        try:
            # Prepare category data
            category_data = prepare_category_data(df, category)
            
            # Split into train/test
            train_data, test_data = create_train_test_split(category_data)
            
            # Train model
            model = train_category_model(train_data, category)
            
            # Evaluate model
            metrics, forecast = evaluate_model(model, train_data, test_data, category)
            
            # Save model
            model_path = save_model(model, category)
            
            # Plot results for this category
            plot_category_forecast(forecast, train_data, test_data, category)
            
            # Store results
            all_models[category] = model
            all_metrics.append(metrics)
            all_forecasts.append(forecast)
            all_test_data.append(test_data)
            
            print(f"✅ Successfully completed {category}")
            
        except Exception as e:
            print(f"❌ Error processing {category}: {str(e)}")
            continue
    
    # Compare all models
    if all_metrics:
        print(f"\n{'#'*80}")
        print("FINAL COMPARISON AND ANALYSIS")
        print(f"{'#'*80}")
        
        df_comparison = compare_category_performance(all_metrics)
        
        # Create interactive comparison
        create_interactive_comparison(df_comparison.to_dict('records'), all_forecasts, all_test_data)
        
        # Save comparison results
        df_comparison.to_csv('category_model_comparison.csv', index=False)
        print(f"\n✅ Comparison results saved to 'category_model_comparison.csv'")
        
        # Final recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        best_category = df_comparison.iloc[0]['category']
        worst_category = df_comparison.iloc[-1]['category']
        avg_rmse = df_comparison['RMSE'].mean()
        
        print(f"🏆 Best performing category: {best_category} (RMSE: {df_comparison.iloc[0]['RMSE']:.2f})")
        print(f"⚠️  Worst performing category: {worst_category} (RMSE: {df_comparison.iloc[-1]['RMSE']:.2f})")
        print(f"📊 Average RMSE across categories: {avg_rmse:.2f}")
        
        if df_comparison['R2'].min() > 0.7:
            print(f"✅ All models show good performance (R² > 0.7)")
        else:
            poor_models = df_comparison[df_comparison['R2'] < 0.7]['category'].tolist()
            print(f"⚠️  Consider improving models for: {poor_models}")
        
        print(f"\n📈 Next steps:")
        print(f"   1. Fine-tune parameters for underperforming categories")
        print(f"   2. Consider adding custom seasonalities or holidays")
        print(f"   3. Implement automated retraining pipeline")
        print(f"   4. Deploy models for production forecasting")
        
    else:
        print("❌ No models were successfully trained!")
    
    return all_models, all_metrics

if __name__ == "__main__":
    models, metrics = main()
