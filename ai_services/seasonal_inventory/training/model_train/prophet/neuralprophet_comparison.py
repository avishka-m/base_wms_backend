"""
NeuralProphet vs Prophet Comparison for Demand Forecasting
Advanced neural network approach vs traditional statistical approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import seaborn as sns

warnings.filterwarnings('ignore')

class DemandForecastingComparison:
    """
    Compare Prophet vs NeuralProphet for demand forecasting
    """
    
    def __init__(self):
        self.results = {}
        self.models = {'prophet': {}, 'neuralprophet': {}}
        
    def load_and_prepare_data(self):
        """Load clean enhanced dataset"""
        print("="*80)
        print("LOADING CLEAN ENHANCED DATASET")
        print("="*80)
        
        df = pd.read_csv('daily_demand_clean_enhanced.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Date range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"🏷️ Categories: {df['category'].unique()}")
        print(f"🔢 Features: {list(df.columns)}")
        
        # Aggregate by category
        category_data = df.groupby(['category', 'ds']).agg({
            'y': 'sum',
            'is_weekend': 'first',
            'month': 'first',
            'quarter': 'first',
            'day_of_week': 'first',
            'season': 'first'
        }).reset_index()
        
        print(f"📈 Category aggregated data: {category_data.shape}")
        
        return category_data
    
    def prepare_neuralprophet_data(self, data):
        """Prepare data specifically for NeuralProphet"""
        
        # NeuralProphet expects specific column names
        np_data = data.copy()
        
        # Add future regressors (external features)
        # NeuralProphet can handle these better than Prophet
        
        # Create seasonal dummy variables
        season_dummies = pd.get_dummies(np_data['season'], prefix='season')
        np_data = pd.concat([np_data, season_dummies], axis=1)
        
        return np_data
    
    def train_prophet_model(self, train_data, category):
        """Train traditional Prophet model"""
        print(f"  🔮 Training Prophet for {category}...")
        
        # Basic Prophet setup
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0
        )
        
        # Add external regressors
        model.add_regressor('is_weekend')
        model.add_regressor('month')
        
        # Fit model
        prophet_data = train_data[['ds', 'y', 'is_weekend', 'month']].copy()
        model.fit(prophet_data)
        
        return model
    
    def train_neuralprophet_model(self, train_data, category):
        """Train NeuralProphet model"""
        print(f"  🧠 Training NeuralProphet for {category}...")
        
        # NeuralProphet with neural networks for non-linear patterns
        model = NeuralProphet(
            growth='linear',  # Can be 'linear' or 'discontinuous'
            n_forecasts=1,    # Single step forecasting
            n_lags=7,         # Use 7 days of historical values
            num_hidden_layers=2,  # Neural network depth
            d_hidden=32,      # Hidden layer size
            learning_rate=1e-3,   # Learning rate
            epochs=100,       # Training epochs
            batch_size=64,    # Batch size
            loss_func='MSE',  # Loss function
            normalize='auto', # Auto-normalize features
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            drop_missing=True,
            impute_missing=True
        )
        
        # Add future regressors (external features)
        model.add_future_regressor('is_weekend')
        model.add_future_regressor('month')
        
        # Fit model (NeuralProphet uses different column names)
        np_data = train_data[['ds', 'y', 'is_weekend', 'month']].copy()
        
        # Train the model
        metrics = model.fit(np_data, freq='D', validation_df=None, progress=None)
        
        return model
    
    def evaluate_model(self, model, test_data, model_type, category):
        """Evaluate model performance"""
        
        if model_type == 'prophet':
            # Prophet evaluation
            future_df = test_data[['ds', 'is_weekend', 'month']].copy()
            forecast = model.predict(future_df)
            predictions = forecast['yhat'].values
            
        else:  # neuralprophet
            # NeuralProphet evaluation
            future_df = test_data[['ds', 'is_weekend', 'month']].copy()
            forecast = model.predict(future_df)
            predictions = forecast['yhat1'].values  # NeuralProphet uses yhat1
        
        # Calculate metrics
        actual = test_data['y'].values
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # MAPE calculation (handle zero values)
        mape = np.mean(np.abs((actual - predictions) / np.maximum(actual, 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': predictions,
            'forecast': forecast
        }
    
    def compare_models_by_category(self, data):
        """Compare Prophet vs NeuralProphet for each category"""
        
        print("\n" + "="*80)
        print("PROPHET vs NEURALPROPHET COMPARISON BY CATEGORY")
        print("="*80)
        
        categories = data['category'].unique()
        comparison_results = {}
        
        for category in categories:
            print(f"\n{'='*60}")
            print(f"CATEGORY: {category.upper()}")
            print(f"{'='*60}")
            
            # Get category data
            cat_data = data[data['category'] == category].copy()
            cat_data = cat_data.sort_values('ds').reset_index(drop=True)
            
            print(f"📊 Data points: {len(cat_data)}")
            print(f"📈 Demand range: {cat_data['y'].min():.0f} to {cat_data['y'].max():.0f}")
            
            # Split data (80/20)
            split_idx = int(len(cat_data) * 0.8)
            train_data = cat_data[:split_idx].copy()
            test_data = cat_data[split_idx:].copy()
            
            print(f"🔄 Train: {len(train_data)} days, Test: {len(test_data)} days")
            
            # Train both models
            prophet_model = self.train_prophet_model(train_data, category)
            neuralprophet_model = self.train_neuralprophet_model(train_data, category)
            
            # Evaluate both models
            prophet_results = self.evaluate_model(prophet_model, test_data, 'prophet', category)
            neuralprophet_results = self.evaluate_model(neuralprophet_model, test_data, 'neuralprophet', category)
            
            # Store results
            comparison_results[category] = {
                'prophet': prophet_results,
                'neuralprophet': neuralprophet_results,
                'test_data': test_data,
                'prophet_model': prophet_model,
                'neuralprophet_model': neuralprophet_model
            }
            
            # Print comparison
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"                    Prophet    NeuralProphet    Improvement")
            print(f"  RMSE:            {prophet_results['rmse']:8.1f}      {neuralprophet_results['rmse']:8.1f}      {((prophet_results['rmse'] - neuralprophet_results['rmse'])/prophet_results['rmse']*100):6.1f}%")
            print(f"  MAE:             {prophet_results['mae']:8.1f}      {neuralprophet_results['mae']:8.1f}      {((prophet_results['mae'] - neuralprophet_results['mae'])/prophet_results['mae']*100):6.1f}%")
            print(f"  R²:              {prophet_results['r2']:8.3f}      {neuralprophet_results['r2']:8.3f}      {((neuralprophet_results['r2'] - prophet_results['r2'])/abs(prophet_results['r2'])*100):6.1f}%")
            print(f"  MAPE:            {prophet_results['mape']:8.1f}%     {neuralprophet_results['mape']:8.1f}%     {((prophet_results['mape'] - neuralprophet_results['mape'])/prophet_results['mape']*100):6.1f}%")
            
            # Determine winner
            prophet_score = (prophet_results['rmse'] + prophet_results['mae'] + prophet_results['mape']) / 3
            neuralprophet_score = (neuralprophet_results['rmse'] + neuralprophet_results['mae'] + neuralprophet_results['mape']) / 3
            
            if neuralprophet_score < prophet_score:
                print(f"  🏆 Winner: NeuralProphet (better overall metrics)")
            else:
                print(f"  🏆 Winner: Prophet (better overall metrics)")
        
        return comparison_results
    
    def create_comparison_visualizations(self, comparison_results):
        """Create comprehensive visualizations"""
        
        print(f"\n📊 Creating comparison visualizations...")
        
        # 1. Performance metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        categories = list(comparison_results.keys())
        metrics = ['rmse', 'mae', 'r2', 'mape']
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            prophet_values = [comparison_results[cat]['prophet'][metric] for cat in categories]
            neuralprophet_values = [comparison_results[cat]['neuralprophet'][metric] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, prophet_values, width, label='Prophet', alpha=0.8)
            bars2 = ax.bar(x + width/2, neuralprophet_values, width, label='NeuralProphet', alpha=0.8)
            
            ax.set_title(f'{name} Comparison')
            ax.set_xlabel('Category')
            ax.set_ylabel(name)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Prophet vs NeuralProphet - Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prophet_vs_neuralprophet_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Forecast visualization for each category
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (category, results) in enumerate(comparison_results.items()):
            ax = axes[idx]
            test_data = results['test_data']
            
            # Plot actual data
            ax.plot(test_data['ds'], test_data['y'], 'k-', label='Actual', linewidth=2, alpha=0.8)
            
            # Plot Prophet forecast
            prophet_pred = results['prophet']['predictions']
            ax.plot(test_data['ds'], prophet_pred, 'b--', label='Prophet', linewidth=2, alpha=0.7)
            
            # Plot NeuralProphet forecast
            neuralprophet_pred = results['neuralprophet']['predictions']
            ax.plot(test_data['ds'], neuralprophet_pred, 'r:', label='NeuralProphet', linewidth=2, alpha=0.7)
            
            ax.set_title(f'{category.title()}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Forecast Comparison - Test Period', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('forecast_comparison_all_categories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_recommendation(self, comparison_results):
        """Generate final recommendation based on results"""
        
        print(f"\n{'='*80}")
        print("FINAL RECOMMENDATION: PROPHET vs NEURALPROPHET")
        print(f"{'='*80}")
        
        # Calculate overall performance
        prophet_wins = 0
        neuralprophet_wins = 0
        total_improvement = {'rmse': 0, 'mae': 0, 'r2': 0, 'mape': 0}
        
        for category, results in comparison_results.items():
            prophet_metrics = results['prophet']
            neuralprophet_metrics = results['neuralprophet']
            
            # Calculate improvement percentages
            rmse_improvement = (prophet_metrics['rmse'] - neuralprophet_metrics['rmse']) / prophet_metrics['rmse'] * 100
            mae_improvement = (prophet_metrics['mae'] - neuralprophet_metrics['mae']) / prophet_metrics['mae'] * 100
            r2_improvement = (neuralprophet_metrics['r2'] - prophet_metrics['r2']) / abs(prophet_metrics['r2']) * 100
            mape_improvement = (prophet_metrics['mape'] - neuralprophet_metrics['mape']) / prophet_metrics['mape'] * 100
            
            total_improvement['rmse'] += rmse_improvement
            total_improvement['mae'] += mae_improvement
            total_improvement['r2'] += r2_improvement
            total_improvement['mape'] += mape_improvement
            
            # Determine winner for this category
            if (rmse_improvement + mae_improvement + mape_improvement) > 0:
                neuralprophet_wins += 1
            else:
                prophet_wins += 1
        
        # Average improvements
        avg_improvements = {k: v / len(comparison_results) for k, v in total_improvement.items()}
        
        print(f"📊 OVERALL RESULTS:")
        print(f"  • Categories where NeuralProphet wins: {neuralprophet_wins}/{len(comparison_results)}")
        print(f"  • Categories where Prophet wins: {prophet_wins}/{len(comparison_results)}")
        print(f"\n📈 AVERAGE IMPROVEMENTS (NeuralProphet vs Prophet):")
        print(f"  • RMSE: {avg_improvements['rmse']:+.1f}%")
        print(f"  • MAE: {avg_improvements['mae']:+.1f}%")
        print(f"  • R²: {avg_improvements['r2']:+.1f}%")
        print(f"  • MAPE: {avg_improvements['mape']:+.1f}%")
        
        # Make recommendation
        if neuralprophet_wins > prophet_wins:
            print(f"\n🏆 RECOMMENDATION: USE NEURALPROPHET")
            print(f"  ✅ Better performance in {neuralprophet_wins}/{len(comparison_results)} categories")
            print(f"  ✅ Handles non-linear patterns better")
            print(f"  ✅ Can leverage neural networks for complex seasonality")
            print(f"  ✅ Better at capturing product interaction effects")
        else:
            print(f"\n🏆 RECOMMENDATION: USE PROPHET")
            print(f"  ✅ Better performance in {prophet_wins}/{len(comparison_results)} categories")
            print(f"  ✅ Simpler and more interpretable")
            print(f"  ✅ Faster training and prediction")
            print(f"  ✅ More stable for production")
        
        print(f"\n🎯 BUSINESS CONSIDERATIONS:")
        if neuralprophet_wins > prophet_wins:
            print(f"  • NeuralProphet advantages:")
            print(f"    - Better accuracy for complex demand patterns")
            print(f"    - Can handle non-linear seasonality")
            print(f"    - Automatic feature learning")
            print(f"    - Better for high-volume, complex products")
            print(f"  • NeuralProphet disadvantages:")
            print(f"    - Longer training time")
            print(f"    - More complex hyperparameter tuning")
            print(f"    - Less interpretable")
            print(f"    - Requires more computational resources")
        else:
            print(f"  • Prophet advantages:")
            print(f"    - Simple and interpretable")
            print(f"    - Fast training and prediction")
            print(f"    - Proven in production")
            print(f"    - Easy hyperparameter tuning")
            print(f"  • Consider NeuralProphet for:")
            print(f"    - High-value product categories")
            print(f"    - Complex seasonal patterns")
            print(f"    - When accuracy is more important than speed")
        
        return avg_improvements

def main():
    """Main execution function"""
    
    # Initialize comparison
    comparison = DemandForecastingComparison()
    
    # Load data
    data = comparison.load_and_prepare_data()
    
    # Compare models
    results = comparison.compare_models_by_category(data)
    
    # Create visualizations
    comparison.create_comparison_visualizations(results)
    
    # Generate recommendation
    improvements = comparison.generate_final_recommendation(results)
    
    return results, improvements

if __name__ == "__main__":
    print("🚀 Starting Prophet vs NeuralProphet comparison...")
    
    try:
        results, improvements = main()
        print(f"\n✅ Comparison complete! Check the generated plots and recommendations.")
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        print("This might be due to NeuralProphet installation or data issues.")
        print("Prophet remains a reliable fallback option.")
