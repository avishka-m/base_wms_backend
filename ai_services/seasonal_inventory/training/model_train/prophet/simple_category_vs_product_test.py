"""
Test if Category Models Can Forecast Individual Products
Simple comparison between category-level and product-specific forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def load_category_params():
    """Load optimized category parameters"""
    try:
        # Load the config file
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

def train_category_model(df, category, params):
    """Train a Prophet model for an entire category"""
    # Aggregate all products in this category
    cat_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
    cat_data = cat_data.sort_values('ds').reset_index(drop=True)
    
    # Add standard parameters
    full_params = params.copy()
    full_params.update({
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'holidays_prior_scale': 10.0
    })
    
    # Train model
    model = Prophet(**full_params)
    model.fit(cat_data)
    
    return model, cat_data

def predict_product_with_category_model(df, product_id, category_model, test_days=30):
    """
    Use category model to predict individual product demand
    Method: Category forecast × Product's historical market share
    """
    # Get product data
    product_data = df[df['product_id'] == product_id].copy()
    category = product_data['category'].iloc[0]
    
    # Split into train/test
    split_idx = len(product_data) - test_days
    train_data = product_data[:split_idx]
    test_data = product_data[split_idx:]
    
    # Calculate product's share of category during training period
    train_end_date = train_data['ds'].max()
    category_train = df[(df['category'] == category) & (df['ds'] <= train_end_date)]
    
    total_category_demand = category_train.groupby('ds')['y'].sum().sum()
    total_product_demand = train_data['y'].sum()
    product_share = total_product_demand / total_category_demand if total_category_demand > 0 else 0
    
    # Predict category demand for test period
    test_dates = test_data[['ds']].copy()
    category_forecast = category_model.predict(test_dates)
    
    # Scale by product share
    product_forecast = category_forecast['yhat'] * product_share
    
    return product_forecast.values, test_data['y'].values, product_share

def train_product_specific_model(df, product_id, category_params):
    """Train a model specifically for one product"""
    product_data = df[df['product_id'] == product_id].copy()
    product_data = product_data[['ds', 'y']].sort_values('ds').reset_index(drop=True)
    
    category = df[df['product_id'] == product_id]['category'].iloc[0]
    
    # Use category parameters as base
    params = category_params[category].copy()
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

def test_category_vs_product_forecasting():
    """Main comparison function"""
    
    print("="*60)
    print("CATEGORY vs PRODUCT MODEL COMPARISON")
    print("="*60)
    
    # Load data
    df = pd.read_csv('daily_demand_by_product_modern.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Load parameters
    category_params = load_category_params()
    
    print(f"Loaded data: {len(df)} rows, {df['product_id'].nunique()} products")
    
    # Train category models
    print("\n🏷️ Training category models...")
    category_models = {}
    category_info = {}
    
    for category in df['category'].unique():
        print(f"  Training {category}...")
        model, cat_data = train_category_model(df, category, category_params[category])
        category_models[category] = model
        category_info[category] = {
            'total_demand': cat_data['y'].sum(),
            'avg_daily': cat_data['y'].mean(),
            'products': df[df['category'] == category]['product_id'].nunique()
        }
    
    # Select test products (2 from each category with good data)
    print("\n📊 Selecting test products...")
    test_products = []
    
    for category in df['category'].unique():
        # Get products with reasonable demand and enough data
        cat_products = df[df['category'] == category].groupby('product_id')['y'].agg(['mean', 'count'])
        suitable = cat_products[(cat_products['mean'] >= 3) & (cat_products['count'] >= 400)]
        
        if len(suitable) >= 2:
            selected = suitable.nlargest(2, 'mean').index.tolist()
            test_products.extend(selected)
            print(f"  {category}: {selected}")
    
    print(f"\nTesting {len(test_products)} products")
    
    # Run comparison
    print("\n🔬 Running forecasting comparison...")
    results = []
    
    for i, product_id in enumerate(test_products):
        print(f"\n[{i+1}/{len(test_products)}] {product_id}")
        
        try:
            product_data = df[df['product_id'] == product_id]
            category = product_data['category'].iloc[0]
            
            if len(product_data) < 100:
                print(f"  ⚠️ Insufficient data: {len(product_data)} days")
                continue
            
            print(f"  Category: {category}")
            print(f"  Data points: {len(product_data)}")
            print(f"  Avg daily demand: {product_data['y'].mean():.1f}")
            
            # Method 1: Category model approach
            category_pred, actual_values, product_share = predict_product_with_category_model(
                df, product_id, category_models[category], test_days=30
            )
            
            # Method 2: Product-specific model
            product_model = train_product_specific_model(df, product_id, category_params)
            
            # Get test data for product model
            split_idx = len(product_data) - 30
            test_dates = product_data[split_idx:][['ds']].copy()
            product_forecast = product_model.predict(test_dates)
            product_pred = product_forecast['yhat'].values
            
            # Calculate metrics
            def calc_metrics(actual, predicted):
                predicted = np.maximum(predicted, 0)  # No negative predictions
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mae = mean_absolute_error(actual, predicted)
                
                # MAPE (handle zeros)
                non_zero = actual != 0
                if non_zero.sum() > 0:
                    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
                else:
                    mape = np.inf
                
                return rmse, mae, mape
            
            cat_rmse, cat_mae, cat_mape = calc_metrics(actual_values, category_pred)
            prod_rmse, prod_mae, prod_mape = calc_metrics(actual_values, product_pred)
            
            # Determine winner
            winner = 'Product' if prod_rmse < cat_rmse else 'Category'
            improvement = abs(prod_rmse - cat_rmse) / max(prod_rmse, cat_rmse) * 100
            
            print(f"  📊 Results:")
            print(f"    Category Model RMSE: {cat_rmse:.2f}")
            print(f"    Product Model RMSE:  {prod_rmse:.2f}")
            print(f"    Winner: {winner} (improvement: {improvement:.1f}%)")
            print(f"    Product share: {product_share:.3f}")
            
            # Store results
            results.append({
                'product_id': product_id,
                'category': category,
                'product_share': product_share,
                'avg_demand': product_data['y'].mean(),
                'category_rmse': cat_rmse,
                'product_rmse': prod_rmse,
                'category_mae': cat_mae,
                'product_mae': prod_mae,
                'winner': winner,
                'improvement_pct': improvement,
                'actual': actual_values,
                'category_pred': category_pred,
                'product_pred': product_pred
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:80]}")
            continue
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    
    if results:
        total_tests = len(results)
        category_wins = sum(1 for r in results if r['winner'] == 'Category')
        product_wins = sum(1 for r in results if r['winner'] == 'Product')
        
        avg_cat_rmse = np.mean([r['category_rmse'] for r in results])
        avg_prod_rmse = np.mean([r['product_rmse'] for r in results])
        
        print(f"\n📊 Overall Results:")
        print(f"  • Total products tested: {total_tests}")
        print(f"  • Category model wins: {category_wins} ({category_wins/total_tests:.1%})")
        print(f"  • Product model wins: {product_wins} ({product_wins/total_tests:.1%})")
        print(f"  • Average Category RMSE: {avg_cat_rmse:.2f}")
        print(f"  • Average Product RMSE: {avg_prod_rmse:.2f}")
        
        overall_improvement = (avg_cat_rmse - avg_prod_rmse) / avg_cat_rmse * 100
        if overall_improvement > 0:
            print(f"  • Product models improve RMSE by {overall_improvement:.1f}% on average")
        else:
            print(f"  • Category models perform {abs(overall_improvement):.1f}% better on average")
        
        # By category analysis
        print(f"\n📈 Results by Category:")
        category_summary = {}
        for result in results:
            cat = result['category']
            if cat not in category_summary:
                category_summary[cat] = {'total': 0, 'category_wins': 0, 'improvements': []}
            
            category_summary[cat]['total'] += 1
            if result['winner'] == 'Category':
                category_summary[cat]['category_wins'] += 1
            category_summary[cat]['improvements'].append(result['improvement_pct'])
        
        for category, stats in category_summary.items():
            win_rate = stats['category_wins'] / stats['total']
            avg_improvement = np.mean(stats['improvements'])
            print(f"  {category}: {stats['category_wins']}/{stats['total']} category wins ({win_rate:.1%}), avg improvement: {avg_improvement:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if product_wins / total_tests > 0.7:
            print("  ✅ Product-specific models are significantly better")
            print("  ✅ Recommend individual product modeling for higher accuracy")
        elif category_wins / total_tests > 0.7:
            print("  ✅ Category models work well for individual products")
            print("  ✅ Category-level forecasting is sufficient and more efficient")
        else:
            print("  ⚖️ Mixed results - consider hybrid approach:")
            print("     • Use category models for most products (efficiency)")
            print("     • Use product models for high-value products (accuracy)")
        
        # Create simple visualization
        create_comparison_plot(results[:6])  # Plot first 6 products
        
    return results

def create_comparison_plot(results):
    """Create simple comparison plot"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= 6:
            break
        
        ax = axes[i]
        days = range(len(result['actual']))
        
        ax.plot(days, result['actual'], 'o-', label='Actual', alpha=0.8, markersize=4)
        ax.plot(days, result['category_pred'], 's-', label='Category Model', alpha=0.8, markersize=3)
        ax.plot(days, result['product_pred'], '^-', label='Product Model', alpha=0.8, markersize=3)
        
        ax.set_title(f"{result['product_id']}\n{result['category']} | Winner: {result['winner']}")
        ax.set_xlabel('Test Days')
        ax.set_ylabel('Demand')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('category_vs_product_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Comparison plot saved as 'category_vs_product_comparison.png'")

if __name__ == "__main__":
    results = test_category_vs_product_forecasting()
    
    print(f"\n🎯 Analysis complete!")
    print(f"  • Check the generated plot for visual comparison")
    print(f"  • Results show whether category models are sufficient for individual products")
