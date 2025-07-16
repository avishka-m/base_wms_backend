"""
Individual vs Category Model Accuracy Comparison
Compare forecasting accuracy for 5 products from the same category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from datetime import datetime, timedelta
import seaborn as sns

warnings.filterwarnings('ignore')

class ProductVsCategoryComparison:
    """
    Compare individual product models vs category model performance
    """
    
    def __init__(self):
        self.results = {}
        self.individual_models = {}
        self.category_model = None
        
    def load_and_prepare_data(self, category_name='electronics'):
        """Load data and select 5 products from specified category"""
        print("="*80)
        print(f"INDIVIDUAL vs CATEGORY MODEL COMPARISON - {category_name.upper()}")
        print("="*80)
        
        # Load data
        df = pd.read_csv('daily_demand_clean_enhanced.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        
        print(f"📊 Full dataset: {len(df):,} rows")
        
        # Filter to selected category
        category_data = df[df['category'] == category_name].copy()
        print(f"📊 {category_name} data: {len(category_data):,} rows")
        print(f"📦 Products in {category_name}: {category_data['product_id'].nunique()}")
        
        # Select 5 products with highest volume (for meaningful comparison)
        product_volumes = category_data.groupby('product_id')['y'].sum().sort_values(ascending=False)
        selected_products = product_volumes.head(5).index.tolist()
        
        print(f"🎯 Selected 5 highest-volume products:")
        for i, product in enumerate(selected_products, 1):
            volume = product_volumes[product]
            print(f"  {i}. {product}: {volume:.0f} total demand")
        
        # Prepare category aggregated data
        category_agg = category_data.groupby('ds')['y'].sum().reset_index()
        category_agg = category_agg.sort_values('ds').reset_index(drop=True)
        
        # Prepare individual product data
        product_data = {}
        for product in selected_products:
            prod_data = category_data[category_data['product_id'] == product][['ds', 'y']].copy()
            prod_data = prod_data.sort_values('ds').reset_index(drop=True)
            product_data[product] = prod_data
        
        # Calculate market shares for category allocation
        market_shares = {}
        total_category_demand = category_data['y'].sum()
        for product in selected_products:
            product_demand = category_data[category_data['product_id'] == product]['y'].sum()
            market_shares[product] = product_demand / total_category_demand
            
        print(f"\n📊 Market shares:")
        for product, share in market_shares.items():
            print(f"  {product}: {share:.3f} ({share*100:.1f}%)")
        
        return category_agg, product_data, market_shares, selected_products
    
    def train_category_model(self, category_data):
        """Train category-level model"""
        print(f"\n🔮 Training category model...")
        
        # Split data (80/20)
        split_idx = int(len(category_data) * 0.8)
        train_data = category_data[:split_idx].copy()
        test_data = category_data[split_idx:].copy()
        
        print(f"  📅 Training period: {train_data['ds'].min()} to {train_data['ds'].max()}")
        print(f"  📅 Test period: {test_data['ds'].min()} to {test_data['ds'].max()}")
        
        # Train model with optimized parameters (based on previous analysis)
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=10.0
        )
        
        model.fit(train_data)
        
        # Generate forecast for test period
        future_df = test_data[['ds']].copy()
        forecast = model.predict(future_df)
        
        # Calculate metrics
        actual = test_data['y'].values
        predicted = forecast['yhat'].values
        
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
        
        print(f"  ✅ Category model trained successfully")
        print(f"     RMSE: {rmse:.1f}, MAE: {mae:.1f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")
        
        self.category_model = model
        
        return {
            'model': model,
            'train_data': train_data,
            'test_data': test_data,
            'forecast': forecast,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
        }
    
    def train_individual_models(self, product_data):
        """Train individual models for each product"""
        print(f"\n🎯 Training individual product models...")
        
        individual_results = {}
        
        for i, (product_id, data) in enumerate(product_data.items(), 1):
            print(f"\n  [{i}/5] Training model for {product_id}...")
            
            if len(data) < 100:
                print(f"    ❌ Insufficient data: {len(data)} days")
                continue
            
            # Split data (80/20)
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx].copy()
            test_data = data[split_idx:].copy()
            
            try:
                # Train individual model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    changepoint_prior_scale=0.1,
                    seasonality_prior_scale=10.0
                )
                
                model.fit(train_data)
                
                # Generate forecast
                future_df = test_data[['ds']].copy()
                forecast = model.predict(future_df)
                
                # Calculate metrics
                actual = test_data['y'].values
                predicted = forecast['yhat'].values
                
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                r2 = r2_score(actual, predicted)
                mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
                
                individual_results[product_id] = {
                    'model': model,
                    'train_data': train_data,
                    'test_data': test_data,
                    'forecast': forecast,
                    'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
                }
                
                print(f"    ✅ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")
                
                self.individual_models[product_id] = model
                
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
        
        return individual_results
    
    def generate_category_based_forecasts(self, category_results, market_shares, individual_results):
        """Generate product forecasts using category model + market share allocation"""
        print(f"\n📊 Generating category-based product forecasts...")
        
        category_based_results = {}
        
        for product_id in individual_results.keys():
            print(f"  📈 Allocating category forecast to {product_id}...")
            
            # Get category forecast
            category_forecast = category_results['forecast']
            market_share = market_shares[product_id]
            
            # Allocate to product using market share
            product_forecast = category_forecast.copy()
            product_forecast['yhat'] *= market_share
            product_forecast['yhat_lower'] *= market_share
            product_forecast['yhat_upper'] *= market_share
            
            # Get actual test data for this product
            actual_test = individual_results[product_id]['test_data']
            
            # Calculate metrics for category-based approach
            actual = actual_test['y'].values
            predicted = product_forecast['yhat'].values
            
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
            
            category_based_results[product_id] = {
                'forecast': product_forecast,
                'test_data': actual_test,
                'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape},
                'market_share': market_share
            }
            
            print(f"    Market share: {market_share:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        return category_based_results
    
    def create_accuracy_comparison(self, individual_results, category_based_results):
        """Create detailed accuracy comparison"""
        print(f"\n{'='*80}")
        print("ACCURACY COMPARISON: INDIVIDUAL vs CATEGORY MODELS")
        print(f"{'='*80}")
        
        # Create comparison table
        comparison_data = []
        
        for product_id in individual_results.keys():
            individual_metrics = individual_results[product_id]['metrics']
            category_metrics = category_based_results[product_id]['metrics']
            
            # Calculate improvement percentages
            rmse_improvement = ((category_metrics['rmse'] - individual_metrics['rmse']) / category_metrics['rmse']) * 100
            mae_improvement = ((category_metrics['mae'] - individual_metrics['mae']) / category_metrics['mae']) * 100
            r2_improvement = ((individual_metrics['r2'] - category_metrics['r2']) / abs(category_metrics['r2'])) * 100
            mape_improvement = ((category_metrics['mape'] - individual_metrics['mape']) / category_metrics['mape']) * 100
            
            comparison_data.append({
                'Product': product_id,
                'Individual_RMSE': individual_metrics['rmse'],
                'Category_RMSE': category_metrics['rmse'],
                'RMSE_Improvement_%': rmse_improvement,
                'Individual_MAE': individual_metrics['mae'],
                'Category_MAE': category_metrics['mae'],
                'MAE_Improvement_%': mae_improvement,
                'Individual_R2': individual_metrics['r2'],
                'Category_R2': category_metrics['r2'],
                'R2_Improvement_%': r2_improvement,
                'Individual_MAPE': individual_metrics['mape'],
                'Category_MAPE': category_metrics['mape'],
                'MAPE_Improvement_%': mape_improvement
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print detailed comparison
        print(f"{'Product':<20} {'Metric':<8} {'Individual':<12} {'Category':<12} {'Improvement':<12}")
        print("-" * 80)
        
        for _, row in comparison_df.iterrows():
            product = row['Product']
            print(f"{product:<20} {'RMSE':<8} {row['Individual_RMSE']:<12.2f} {row['Category_RMSE']:<12.2f} {row['RMSE_Improvement_%']:>10.1f}%")
            print(f"{'':<20} {'MAE':<8} {row['Individual_MAE']:<12.2f} {row['Category_MAE']:<12.2f} {row['MAE_Improvement_%']:>10.1f}%")
            print(f"{'':<20} {'R²':<8} {row['Individual_R2']:<12.3f} {row['Category_R2']:<12.3f} {row['R2_Improvement_%']:>10.1f}%")
            print(f"{'':<20} {'MAPE':<8} {row['Individual_MAPE']:<12.1f} {row['Category_MAPE']:<12.1f} {row['MAPE_Improvement_%']:>10.1f}%")
            print("-" * 80)
        
        # Summary statistics
        avg_rmse_improvement = comparison_df['RMSE_Improvement_%'].mean()
        avg_mae_improvement = comparison_df['MAE_Improvement_%'].mean()
        avg_r2_improvement = comparison_df['R2_Improvement_%'].mean()
        avg_mape_improvement = comparison_df['MAPE_Improvement_%'].mean()
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Average RMSE improvement: {avg_rmse_improvement:+.1f}%")
        print(f"  Average MAE improvement: {avg_mae_improvement:+.1f}%")
        print(f"  Average R² improvement: {avg_r2_improvement:+.1f}%")
        print(f"  Average MAPE improvement: {avg_mape_improvement:+.1f}%")
        
        # Determine winner
        individual_wins = sum(1 for imp in comparison_df['RMSE_Improvement_%'] if imp > 0)
        category_wins = len(comparison_df) - individual_wins
        
        print(f"\n🏆 WINNER ANALYSIS:")
        print(f"  Individual models win: {individual_wins}/5 products")
        print(f"  Category models win: {category_wins}/5 products")
        
        if individual_wins > category_wins:
            print(f"  🥇 Individual models are better overall")
        elif category_wins > individual_wins:
            print(f"  🥇 Category models are better overall")
        else:
            print(f"  🤝 Tie - both approaches have merit")
        
        return comparison_df
    
    def create_visualizations(self, individual_results, category_based_results):
        """Create comprehensive visualizations"""
        print(f"\n📊 Creating comparison visualizations...")
        
        # 1. Forecast comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        axes = axes.flatten()
        
        for i, product_id in enumerate(list(individual_results.keys())[:5]):
            ax = axes[i]
            
            # Get data
            individual_data = individual_results[product_id]
            category_data = category_based_results[product_id]
            
            test_data = individual_data['test_data']
            individual_forecast = individual_data['forecast']
            category_forecast = category_data['forecast']
            
            # Plot actual data
            ax.plot(test_data['ds'], test_data['y'], 'k-', label='Actual', linewidth=2, alpha=0.8)
            
            # Plot individual forecast
            ax.plot(test_data['ds'], individual_forecast['yhat'], 'b--', 
                   label='Individual Model', linewidth=2, alpha=0.7)
            
            # Plot category-based forecast
            ax.plot(test_data['ds'], category_forecast['yhat'], 'r:', 
                   label='Category Model', linewidth=2, alpha=0.7)
            
            # Add confidence intervals
            ax.fill_between(test_data['ds'], 
                           individual_forecast['yhat_lower'], 
                           individual_forecast['yhat_upper'],
                           alpha=0.1, color='blue')
            ax.fill_between(test_data['ds'], 
                           category_forecast['yhat_lower'], 
                           category_forecast['yhat_upper'],
                           alpha=0.1, color='red')
            
            # Styling
            individual_rmse = individual_data['metrics']['rmse']
            category_rmse = category_data['metrics']['rmse']
            
            ax.set_title(f'{product_id}\nIndividual RMSE: {individual_rmse:.2f}, Category RMSE: {category_rmse:.2f}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.suptitle('Individual vs Category Model Forecasts - Test Period', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('individual_vs_category_forecasts.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Metrics comparison bar chart
        products = list(individual_results.keys())
        metrics = ['RMSE', 'MAE', 'MAPE']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            individual_values = [individual_results[p]['metrics'][metric.lower()] for p in products]
            category_values = [category_based_results[p]['metrics'][metric.lower()] for p in products]
            
            x = np.arange(len(products))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, individual_values, width, label='Individual', alpha=0.8)
            bars2 = ax.bar(x + width/2, category_values, width, label='Category', alpha=0.8)
            
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Products')
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels([p.split('_')[-1] for p in products], rotation=45)
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
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comparison(self, category_name='electronics'):
        """Run complete comparison analysis"""
        
        # Load and prepare data
        category_agg, product_data, market_shares, selected_products = self.load_and_prepare_data(category_name)
        
        # Train category model
        category_results = self.train_category_model(category_agg)
        
        # Train individual models
        individual_results = self.train_individual_models(product_data)
        
        # Generate category-based forecasts
        category_based_results = self.generate_category_based_forecasts(
            category_results, market_shares, individual_results)
        
        # Create accuracy comparison
        comparison_df = self.create_accuracy_comparison(individual_results, category_based_results)
        
        # Create visualizations
        self.create_visualizations(individual_results, category_based_results)
        
        return {
            'individual_results': individual_results,
            'category_based_results': category_based_results,
            'comparison_df': comparison_df,
            'selected_products': selected_products
        }

def main():
    """Main execution function"""
    
    print("🚀 Starting Individual vs Category Model Comparison...")
    
    # Initialize comparison
    comparison = ProductVsCategoryComparison()
    
    # Run comparison (you can change category here)
    results = comparison.run_comparison(category_name='electronics')
    
    print(f"\n✅ Comparison complete!")
    print(f"📊 Results show the accuracy difference between:")
    print(f"   - Individual product models (trained on each product's data)")
    print(f"   - Category model allocation (category forecast × market share)")
    
    return results

if __name__ == "__main__":
    results = main()
