"""
Category vs Product-Level Model Comparison
Test if category-level models can effectively forecast individual products
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import seaborn as sns
from typing import Dict, List, Tuple
import time

warnings.filterwarnings('ignore')

class CategoryVsProductComparison:
    """
    Compare forecasting accuracy between category-level and product-level models
    """
    
    def __init__(self):
        self.category_models = {}
        self.product_models = {}
        self.category_params = self._load_optimized_params()
        self.results = {}
        
    def _load_optimized_params(self) -> Dict:
        """Load optimized parameters from cross-validation results"""
        try:
            exec(open('prophet_cv_optimized_config.py').read(), globals())
            return CATEGORY_HYPERPARAMETERS
        except:
            print("⚠️ Using default parameters (CV config not found)")
            return {
                'books_media': {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'clothing': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 1.0, 'seasonality_mode': 'additive'},
                'electronics': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 50.0, 'seasonality_mode': 'additive'},
                'health_beauty': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'home_garden': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'},
                'sports_outdoors': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'seasonality_mode': 'additive'}
            }
    
    def train_category_models(self, df: pd.DataFrame) -> None:
        """
        Train one model per category using aggregated data
        """
        print("🏷️ Training category-level models...")
        
        for category in df['category'].unique():
            print(f"  Training {category} model...")
            
            # Aggregate all products in category by date
            cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
            cat_data = cat_data.sort_values('ds').reset_index(drop=True)
            
            # Get optimized parameters for this category
            params = self.category_params[category].copy()
            params.update({
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'holidays_prior_scale': 10.0
            })
            
            # Train model
            model = Prophet(**params)
            model.fit(cat_data)
            
            self.category_models[category] = {
                'model': model,
                'total_demand': cat_data['y'].sum(),
                'avg_daily_demand': cat_data['y'].mean(),
                'data_points': len(cat_data)
            }
        
        print(f"✅ Trained {len(self.category_models)} category models")
    
    def predict_product_with_category_model(self, df: pd.DataFrame, product_id: str, 
                                          test_periods: int = 30) -> Tuple[np.ndarray, Dict]:
        """
        Predict individual product using category-level model
        
        Method: Use category model forecast + product's historical share
        """
        # Get product info
        product_data = df[df['product_id'] == product_id].copy()
        category = product_data['category'].iloc[0]
        
        # Split data for testing
        split_idx = len(product_data) - test_periods
        historical_data = product_data[:split_idx]
        
        # Calculate product's share of category demand during historical period
        cat_historical = df[(df['category'] == category) & 
                           (df['ds'] <= historical_data['ds'].max())]
        cat_total = cat_historical.groupby('ds')['y'].sum()
        product_historical = historical_data.set_index('ds')['y']
        
        # Calculate average share (handling missing dates)
        aligned_data = pd.DataFrame({
            'category_total': cat_total,
            'product_demand': product_historical
        }).fillna(0)
        
        total_cat_demand = aligned_data['category_total'].sum()
        total_product_demand = aligned_data['product_demand'].sum()
        product_share = total_product_demand / total_cat_demand if total_cat_demand > 0 else 0
        
        # Get category model
        category_model = self.category_models[category]['model']
        
        # Predict category demand for test period
        future_dates = pd.date_range(
            start=historical_data['ds'].max() + pd.Timedelta(days=1),
            periods=test_periods,
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        category_forecast = category_model.predict(future_df)
        
        # Calculate product forecast as share of category forecast
        product_forecast = category_forecast['yhat'] * product_share
        
        # Metadata
        metadata = {
            'category': category,
            'product_share': product_share,
            'historical_avg': historical_data['y'].mean(),
            'category_forecast_avg': category_forecast['yhat'].mean(),
            'method': 'category_model_with_share'
        }
        
        return product_forecast.values, metadata
    
    def train_product_specific_model(self, df: pd.DataFrame, product_id: str) -> Prophet:
        """
        Train a model specifically for one product
        """
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        
        category = df[df['product_id'] == product_id]['category'].iloc[0]
        
        # Use category-optimized parameters but adjust for product characteristics
        params = self.category_params[category].copy()
        
        # Analyze product patterns for parameter adjustment
        avg_demand = product_data['y'].mean()
        cv = product_data['y'].std() / avg_demand if avg_demand > 0 else 0
        
        # Adjust parameters based on product characteristics
        if avg_demand < 5:  # Low volume products
            params['seasonality_prior_scale'] = max(0.01, params['seasonality_prior_scale'] * 0.5)
        
        if cv > 0.8:  # High variability products
            params['changepoint_prior_scale'] = min(1.0, params['changepoint_prior_scale'] * 1.5)
        
        params.update({
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'holidays_prior_scale': 10.0
        })
        
        # Train model
        model = Prophet(**params)
        model.fit(product_data)
        
        return model
    
    def compare_forecasting_approaches(self, df: pd.DataFrame, test_products: List[str], 
                                     test_periods: int = 30) -> Dict:
        """
        Compare category vs product-level forecasting for selected products
        """
        print(f"\n🔬 Comparing forecasting approaches for {len(test_products)} products...")
        print(f"   Test period: {test_periods} days")
        
        comparison_results = {
            'product_results': {},
            'summary_metrics': {},
            'category_breakdown': {}
        }
        
        for i, product_id in enumerate(test_products):
            print(f"\n[{i+1}/{len(test_products)}] Testing {product_id}")
            
            try:
                # Get product data
                product_data = df[df['product_id'] == product_id].copy()
                product_data = product_data.sort_values('ds').reset_index(drop=True)
                category = product_data['category'].iloc[0]
                
                if len(product_data) < test_periods + 60:  # Need enough data
                    print(f"  ⚠️ Insufficient data: {len(product_data)} days")
                    continue
                
                # Split data
                split_idx = len(product_data) - test_periods
                train_data = product_data[:split_idx]
                test_data = product_data[split_idx:]
                actual_values = test_data['y'].values
                
                print(f"  Category: {category}")
                print(f"  Train: {len(train_data)} days, Test: {len(test_data)} days")
                print(f"  Avg demand: {product_data['y'].mean():.1f}")
                
                # Method 1: Category model + product share
                category_pred, category_meta = self.predict_product_with_category_model(
                    df, product_id, test_periods
                )
                
                # Method 2: Product-specific model
                print(f"  Training product-specific model...")
                product_model = self.train_product_specific_model(df, product_id)
                
                # Predict with product model
                future_dates = test_data[['ds']].copy()
                product_forecast = product_model.predict(future_dates)
                product_pred = product_forecast['yhat'].values
                
                # Calculate metrics for both approaches
                def calculate_metrics(actual, predicted, method_name):
                    # Handle negative predictions
                    predicted = np.maximum(predicted, 0)
                    
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    mae = mean_absolute_error(actual, predicted)
                    
                    # MAPE with handling for zero values
                    non_zero_mask = actual != 0
                    if non_zero_mask.sum() > 0:
                        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
                    else:
                        mape = np.inf
                    
                    # R² score
                    try:
                        r2 = r2_score(actual, predicted)
                    except:
                        r2 = -np.inf
                    
                    return {
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'r2': r2,
                        'method': method_name,
                        'mean_actual': actual.mean(),
                        'mean_predicted': predicted.mean()
                    }
                
                category_metrics = calculate_metrics(actual_values, category_pred, 'category_model')
                product_metrics = calculate_metrics(actual_values, product_pred, 'product_model')
                
                # Determine winner
                winner = 'product_model' if product_metrics['rmse'] < category_metrics['rmse'] else 'category_model'
                improvement = abs(product_metrics['rmse'] - category_metrics['rmse']) / max(product_metrics['rmse'], category_metrics['rmse']) * 100
                
                print(f"  📊 Results:")
                print(f"    Category Model RMSE: {category_metrics['rmse']:.2f}")
                print(f"    Product Model RMSE:  {product_metrics['rmse']:.2f}")
                print(f"    Winner: {winner} (improvement: {improvement:.1f}%)")
                
                # Store results
                comparison_results['product_results'][product_id] = {
                    'category': category,
                    'product_share': category_meta['product_share'],
                    'category_metrics': category_metrics,
                    'product_metrics': product_metrics,
                    'winner': winner,
                    'improvement_pct': improvement,
                    'actual_values': actual_values,
                    'category_predictions': category_pred,
                    'product_predictions': product_pred,
                    'test_period': test_periods
                }
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
                continue
        
        # Calculate summary statistics
        self._calculate_summary_metrics(comparison_results)
        
        return comparison_results
    
    def _calculate_summary_metrics(self, results: Dict) -> None:
        """Calculate overall summary metrics"""
        product_results = results['product_results']
        
        if not product_results:
            return
        
        # Overall statistics
        total_products = len(product_results)
        category_wins = sum(1 for r in product_results.values() if r['winner'] == 'category_model')
        product_wins = sum(1 for r in product_results.values() if r['winner'] == 'product_model')
        
        # Average improvements
        category_rmses = [r['category_metrics']['rmse'] for r in product_results.values()]
        product_rmses = [r['product_metrics']['rmse'] for r in product_results.values()]
        
        avg_category_rmse = np.mean(category_rmses)
        avg_product_rmse = np.mean(product_rmses)
        
        # By category breakdown
        category_breakdown = {}
        for product_id, result in product_results.items():
            category = result['category']
            if category not in category_breakdown:
                category_breakdown[category] = {
                    'total_products': 0,
                    'category_wins': 0,
                    'product_wins': 0,
                    'avg_improvement': []
                }
            
            category_breakdown[category]['total_products'] += 1
            if result['winner'] == 'category_model':
                category_breakdown[category]['category_wins'] += 1
            else:
                category_breakdown[category]['product_wins'] += 1
            
            category_breakdown[category]['avg_improvement'].append(result['improvement_pct'])
        
        # Finalize category breakdown
        for category in category_breakdown:
            improvements = category_breakdown[category]['avg_improvement']
            category_breakdown[category]['avg_improvement'] = np.mean(improvements)
        
        results['summary_metrics'] = {
            'total_products': total_products,
            'category_model_wins': category_wins,
            'product_model_wins': product_wins,
            'category_win_rate': category_wins / total_products,
            'product_win_rate': product_wins / total_products,
            'avg_category_rmse': avg_category_rmse,
            'avg_product_rmse': avg_product_rmse,
            'overall_improvement': (avg_category_rmse - avg_product_rmse) / avg_category_rmse * 100
        }
        
        results['category_breakdown'] = category_breakdown
    
    def create_comparison_plots(self, results: Dict, max_products: int = 8) -> None:
        """Create visualization comparing the two approaches"""
        
        product_results = results['product_results']
        if not product_results:
            print("No results to plot")
            return
        
        # Select top products for detailed plotting
        selected_products = list(product_results.keys())[:max_products]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: RMSE comparison
        ax1 = axes[0, 0]
        products = []
        category_rmses = []
        product_rmses = []
        
        for product_id in selected_products:
            result = product_results[product_id]
            products.append(product_id.split('_')[-1])  # Short name
            category_rmses.append(result['category_metrics']['rmse'])
            product_rmses.append(result['product_metrics']['rmse'])
        
        x = np.arange(len(products))
        width = 0.35
        
        ax1.bar(x - width/2, category_rmses, width, label='Category Model', alpha=0.8)
        ax1.bar(x + width/2, product_rmses, width, label='Product Model', alpha=0.8)
        ax1.set_xlabel('Products')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Comparison: Category vs Product Models')
        ax1.set_xticks(x)
        ax1.set_xticklabels(products, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win rate by category
        ax2 = axes[0, 1]
        category_breakdown = results['category_breakdown']
        categories = list(category_breakdown.keys())
        category_win_rates = [category_breakdown[cat]['category_wins'] / category_breakdown[cat]['total_products'] 
                             for cat in categories]
        
        ax2.bar(categories, category_win_rates, alpha=0.8, color='skyblue')
        ax2.set_ylabel('Category Model Win Rate')
        ax2.set_title('Category Model Performance by Category')
        ax2.set_xticklabels(categories, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax2.legend()
        
        # Plot 3: Sample forecast comparison
        ax3 = axes[1, 0]
        if selected_products:
            sample_product = selected_products[0]
            result = product_results[sample_product]
            
            days = range(len(result['actual_values']))
            ax3.plot(days, result['actual_values'], 'o-', label='Actual', alpha=0.8)
            ax3.plot(days, result['category_predictions'], 's-', label='Category Model', alpha=0.8)
            ax3.plot(days, result['product_predictions'], '^-', label='Product Model', alpha=0.8)
            
            ax3.set_xlabel('Test Days')
            ax3.set_ylabel('Demand')
            ax3.set_title(f'Sample Forecast Comparison: {sample_product}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Improvement distribution
        ax4 = axes[1, 1]
        improvements = [result['improvement_pct'] for result in product_results.values()]
        winners = [result['winner'] for result in product_results.values()]
        
        # Color by winner
        colors = ['red' if w == 'category_model' else 'blue' for w in winners]
        ax4.scatter(range(len(improvements)), improvements, c=colors, alpha=0.6)
        ax4.set_xlabel('Product Index')
        ax4.set_ylabel('Improvement %')
        ax4.set_title('Model Improvement Distribution')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.6, label='Category Model Wins'),
                          Patch(facecolor='blue', alpha=0.6, label='Product Model Wins')]
        ax4.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig('category_vs_product_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_summary(self, results: Dict) -> None:
        """Print comprehensive summary of results"""
        
        print("\n" + "="*80)
        print("CATEGORY vs PRODUCT MODEL COMPARISON RESULTS")
        print("="*80)
        
        summary = results['summary_metrics']
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"  • Total products tested: {summary['total_products']}")
        print(f"  • Category model wins: {summary['category_model_wins']} ({summary['category_win_rate']:.1%})")
        print(f"  • Product model wins: {summary['product_model_wins']} ({summary['product_win_rate']:.1%})")
        print(f"  • Average Category RMSE: {summary['avg_category_rmse']:.2f}")
        print(f"  • Average Product RMSE: {summary['avg_product_rmse']:.2f}")
        
        if summary['overall_improvement'] > 0:
            print(f"  • Product models improve RMSE by {summary['overall_improvement']:.1f}% on average")
        else:
            print(f"  • Category models perform {abs(summary['overall_improvement']):.1f}% better on average")
        
        print(f"\n📈 PERFORMANCE BY CATEGORY:")
        category_breakdown = results['category_breakdown']
        
        for category, stats in category_breakdown.items():
            win_rate = stats['category_wins'] / stats['total_products']
            print(f"  {category.upper()}:")
            print(f"    • Products tested: {stats['total_products']}")
            print(f"    • Category model wins: {stats['category_wins']} ({win_rate:.1%})")
            print(f"    • Average improvement: {stats['avg_improvement']:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if summary['product_win_rate'] > 0.7:
            print("  ✅ Product-specific models are significantly better")
            print("  ✅ Invest in individual product modeling")
        elif summary['category_win_rate'] > 0.7:
            print("  ✅ Category models are sufficient for most products")
            print("  ✅ Category-level forecasting is recommended for efficiency")
        else:
            print("  ⚖️ Mixed results - consider hybrid approach:")
            print("     • Use category models for most products")
            print("     • Use product models for high-value or unique products")

def run_comparison_analysis():
    """
    Main function to run the complete comparison analysis
    """
    print("🔍 Starting Category vs Product Model Comparison...")
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Initialize comparison system
    comparison = CategoryVsProductComparison()
    
    # Train category models
    comparison.train_category_models(df)
    
    # Select test products (diverse sample from each category)
    print("\n📋 Selecting test products...")
    test_products = []
    
    for category in df['category'].unique():
        # Get products with reasonable demand (not too low, not too high)
        cat_products = df[df['category'] == category].groupby('product_id')['y'].agg(['mean', 'count'])
        
        # Filter: average demand between 5-50, at least 300 days of data
        suitable_products = cat_products[
            (cat_products['mean'] >= 5) & 
            (cat_products['mean'] <= 50) & 
            (cat_products['count'] >= 500)
        ]
        
        if len(suitable_products) > 0:
            # Take top 3 products by demand from each category
            selected = suitable_products.nlargest(3, 'mean').index.tolist()
            test_products.extend(selected)
    
    print(f"Selected {len(test_products)} products for testing")
    
    # Run comparison
    results = comparison.compare_forecasting_approaches(df, test_products, test_periods=30)
    
    # Print detailed summary
    comparison.print_detailed_summary(results)
    
    # Create visualizations
    print(f"\n📊 Creating comparison plots...")
    comparison.create_comparison_plots(results)
    
    return comparison, results

if __name__ == "__main__":
    comparison, results = run_comparison_analysis()
    
    print(f"\n🎯 Analysis complete! Check the generated plots and summary.")
    print(f"   • category_vs_product_comparison.png - Visual comparison")
    print(f"   • Use results dictionary for detailed analysis")
