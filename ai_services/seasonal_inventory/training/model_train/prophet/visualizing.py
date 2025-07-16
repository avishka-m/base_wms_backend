import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_time_features(df):
    """
    Create comprehensive time-based features for Prophet modeling
    Since patterns are consistent across products, same features work for all categories
    """
    print("Creating time-based features...")
    
    # Ensure ds is datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Weekend indicator (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['ds'].dt.dayofweek.isin([5, 6])).astype(int)
    
    # End of week (Friday)
    df['is_end_of_week'] = (df['ds'].dt.dayofweek == 4).astype(int)
    
    # End of month (last day of month)
    df['is_end_of_month'] = df['ds'].dt.is_month_end.astype(int)
    
    # Last 3 days of month (broader end-of-month effect)
    df['days_until_month_end'] = (df['ds'].dt.days_in_month - df['ds'].dt.day)
    df['is_last_3_days_month'] = (df['days_until_month_end'] <= 2).astype(int)
    
    # Beginning of month (first 3 days)
    df['is_beginning_of_month'] = (df['ds'].dt.day <= 3).astype(int)
    
    # Middle of month
    df['is_middle_of_month'] = ((df['ds'].dt.day >= 14) & (df['ds'].dt.day <= 16)).astype(int)
    
    # Payday periods (typically 15th and end of month)
    df['is_payday_period'] = ((df['ds'].dt.day.isin([14, 15, 16])) | 
                              (df['days_until_month_end'] <= 2)).astype(int)
    
    # Holiday season (major shopping periods)
    df['is_holiday_season'] = df['ds'].dt.month.isin([11, 12]).astype(int)  # Nov-Dec
    df['is_back_to_school'] = df['ds'].dt.month.isin([8, 9]).astype(int)    # Aug-Sep
    
    # Specific month effects
    df['is_january'] = (df['ds'].dt.month == 1).astype(int)  # New Year effect
    df['is_december'] = (df['ds'].dt.month == 12).astype(int)  # Christmas effect
    
    # Quarter effects
    df['is_q4'] = (df['ds'].dt.quarter == 4).astype(int)  # Q4 business effect
    
    print("✅ Time features created successfully!")
    return df

def prepare_enhanced_prophet_data(df, category):
    """Prepare data for a category with enhanced features"""
    print(f"\nPreparing enhanced data for category: {category}")
    
    # Add time features to the dataframe
    df = create_time_features(df)
    
    # Filter for the category
    category_df = df[df['category'] == category].copy()
    
    # Aggregate demand by date but keep all the features
    demand_agg = category_df.groupby('ds')['y'].sum().reset_index()
    
    # Get the features (they should be the same for each date)
    feature_cols = [
        'is_weekend', 'is_end_of_week', 'is_end_of_month', 'is_last_3_days_month',
        'is_beginning_of_month', 'is_middle_of_month', 'is_payday_period',
        'is_holiday_season', 'is_back_to_school', 'is_january', 'is_december', 'is_q4'
    ]
    
    features_agg = category_df.groupby('ds')[feature_cols].first().reset_index()
    
    # Merge demand with features
    category_data = pd.merge(demand_agg, features_agg, on='ds')
    category_data = category_data.sort_values('ds').reset_index(drop=True)
    
    print(f"Category data shape: {category_data.shape}")
    print(f"Date range: {category_data['ds'].min()} to {category_data['ds'].max()}")
    print(f"Average daily demand: {category_data['y'].mean():.2f}")
    print(f"Features available: {len(feature_cols)}")
    
    return category_data, feature_cols

def load_and_prepare_data(file_path):
    """Load and prepare data for Prophet modeling"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First few rows:\n{df.head()}")
    
    return df

def prepare_prophet_data(df, date_col, value_col):
    """Prepare data in Prophet format (ds, y)"""
    # Create a copy of the dataframe with only the required columns
    prophet_df = df[[date_col, value_col]].copy()
    # Rename columns to Prophet's required format: 'ds' for dates, 'y' for values
    prophet_df.columns = ['ds', 'y']
    
    # Convert date column to datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Remove any missing values
    prophet_df = prophet_df.dropna()
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Split data into train and test (80% train, 20% test)
    split_index = int(len(prophet_df) * 0.8)
    train_data = prophet_df[:split_index].copy()
    test_data = prophet_df[split_index:].copy()
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train date range: {train_data['ds'].min()} to {train_data['ds'].max()}")
    print(f"Test date range: {test_data['ds'].min()} to {test_data['ds'].max()}")
    
    print(f"Prophet data prepared. Total shape: {prophet_df.shape}")
    print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    
    return prophet_df, train_data, test_data

def create_prophet_model(df, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
    """Create and fit Prophet model"""
    print("Creating Prophet model...")
    
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode='multiplicative'
    )
    
    print("Fitting model...")
    model.fit(df)
    print("Model fitted successfully!")
    
    return model

def create_enhanced_prophet_model(train_data, feature_cols, category, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
    """Create and fit Enhanced Prophet model with regressors"""
    print(f"Creating Enhanced Prophet model for {category}...")
    
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        seasonality_mode='additive'  # Since patterns are consistent, additive works well
    )
    
    # Add all regressors (same for all categories since patterns are consistent)
    print("Adding regressors:")
    for feature in feature_cols:
        if feature in train_data.columns:
            model.add_regressor(feature)
            print(f"  ✓ {feature}")
    
    print("Fitting enhanced model...")
    model.fit(train_data[['ds', 'y'] + feature_cols])
    print("Enhanced model fitted successfully!")
    
    return model

def compare_basic_vs_enhanced_models(df, category):
    """Compare basic Prophet vs enhanced Prophet with regressors for a category"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    print(f"\n{'='*60}")
    print(f"COMPARING MODELS FOR: {category.upper()}")
    print(f"{'='*60}")
    
    # Prepare enhanced data
    category_data, feature_cols = prepare_enhanced_prophet_data(df, category)
    
    # Split data
    split_index = int(len(category_data) * 0.8)
    train_data = category_data[:split_index].copy()
    test_data = category_data[split_index:].copy()
    
    print(f"Train data: {len(train_data)} days")
    print(f"Test data: {len(test_data)} days")
    
    # 1. BASIC PROPHET MODEL
    print(f"\n--- BASIC PROPHET MODEL ---")
    basic_model = create_prophet_model(train_data[['ds', 'y']])
    
    basic_future = basic_model.make_future_dataframe(periods=len(test_data))
    basic_forecast = basic_model.predict(basic_future)
    
    # Basic evaluation
    test_start_idx = len(train_data)
    basic_predictions = basic_forecast['yhat'][test_start_idx:].values
    actual = test_data['y'].values
    
    basic_mae = mean_absolute_error(actual, basic_predictions)
    basic_rmse = np.sqrt(mean_squared_error(actual, basic_predictions))
    basic_r2 = r2_score(actual, basic_predictions)
    
    print(f"Basic Model Performance:")
    print(f"  MAE:  {basic_mae:.2f}")
    print(f"  RMSE: {basic_rmse:.2f}")
    print(f"  R²:   {basic_r2:.4f}")
    
    # 2. ENHANCED PROPHET MODEL
    print(f"\n--- ENHANCED PROPHET MODEL ---")
    enhanced_model = create_enhanced_prophet_model(train_data, feature_cols, category)
    
    # Create enhanced future with features
    enhanced_future = pd.concat([train_data, test_data], ignore_index=True)
    enhanced_future = enhanced_future[['ds'] + feature_cols]
    
    enhanced_forecast = enhanced_model.predict(enhanced_future)
    
    # Enhanced evaluation
    enhanced_predictions = enhanced_forecast['yhat'][test_start_idx:].values
    
    enhanced_mae = mean_absolute_error(actual, enhanced_predictions)
    enhanced_rmse = np.sqrt(mean_squared_error(actual, enhanced_predictions))
    enhanced_r2 = r2_score(actual, enhanced_predictions)
    
    print(f"Enhanced Model Performance:")
    print(f"  MAE:  {enhanced_mae:.2f}")
    print(f"  RMSE: {enhanced_rmse:.2f}")
    print(f"  R²:   {enhanced_r2:.4f}")
    
    # 3. COMPARISON
    mae_improvement = ((basic_mae - enhanced_mae) / basic_mae) * 100
    rmse_improvement = ((basic_rmse - enhanced_rmse) / basic_rmse) * 100
    r2_improvement = enhanced_r2 - basic_r2
    
    print(f"\n--- IMPROVEMENT ANALYSIS ---")
    print(f"MAE improvement:  {mae_improvement:.1f}%")
    print(f"RMSE improvement: {rmse_improvement:.1f}%")
    print(f"R² improvement:   {r2_improvement:.4f}")
    
    if rmse_improvement > 5:
        print("✅ Enhanced model shows significant improvement!")
    elif rmse_improvement > 0:
        print("🟡 Enhanced model shows modest improvement")
    else:
        print("⚠️  Basic model performed better")
    
    # 4. FEATURE IMPORTANCE ANALYSIS
    print(f"\n--- FEATURE IMPORTANCE ---")
    try:
        # Get feature coefficients from forecast components
        feature_effects = {}
        for feature in feature_cols:
            if feature in enhanced_forecast.columns:
                avg_effect = enhanced_forecast[feature].abs().mean()
                feature_effects[feature] = avg_effect
        
        # Sort by importance
        sorted_features = sorted(feature_effects.items(), key=lambda x: x[1], reverse=True)
        
        print("Top features by average absolute effect:")
        for i, (feature, effect) in enumerate(sorted_features[:8], 1):
            print(f"  {i:2d}. {feature:25s}: {effect:.3f}")
            
    except Exception as e:
        print(f"Could not analyze feature importance: {str(e)}")
    
    return {
        'basic': {'mae': basic_mae, 'rmse': basic_rmse, 'r2': basic_r2},
        'enhanced': {'mae': enhanced_mae, 'rmse': enhanced_rmse, 'r2': enhanced_r2},
        'improvement': {'mae': mae_improvement, 'rmse': rmse_improvement, 'r2': r2_improvement}
    }

def train_all_enhanced_category_models(df):
    """Train enhanced models for all 6 categories"""
    print("="*80)
    print("TRAINING ENHANCED MODELS FOR ALL CATEGORIES")
    print("="*80)
    
    categories = df['category'].unique()
    print(f"Categories: {categories}")
    
    all_results = {}
    all_models = {}
    
    for i, category in enumerate(categories, 1):
        print(f"\n{'#'*60}")
        print(f"CATEGORY {i}/{len(categories)}: {category.upper()}")
        print(f"{'#'*60}")
        
        try:
            # Compare models for this category
            results = compare_basic_vs_enhanced_models(df, category)
            all_results[category] = results
            
            # Train final enhanced model on full data for deployment
            print(f"\n--- TRAINING FINAL ENHANCED MODEL ---")
            category_data, feature_cols = prepare_enhanced_prophet_data(df, category)
            final_model = create_enhanced_prophet_model(category_data, feature_cols, category)
            
            all_models[category] = {
                'model': final_model,
                'features': feature_cols,
                'performance': results
            }
            
            print(f"✅ Successfully completed {category}")
            
        except Exception as e:
            print(f"❌ Error processing {category}: {str(e)}")
            continue
    
    # Final summary
    print(f"\n{'#'*80}")
    print("FINAL SUMMARY - ENHANCED vs BASIC MODELS")
    print(f"{'#'*80}")
    
    summary_data = []
    for category, results in all_results.items():
        summary_data.append({
            'Category': category,
            'Basic_RMSE': results['basic']['rmse'],
            'Enhanced_RMSE': results['enhanced']['rmse'],
            'RMSE_Improvement_%': results['improvement']['rmse'],
            'Enhanced_R2': results['enhanced']['r2']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    avg_improvement = summary_df['RMSE_Improvement_%'].mean()
    print(f"\n📊 OVERALL RESULTS:")
    print(f"Average RMSE improvement: {avg_improvement:.1f}%")
    
    if avg_improvement > 5:
        print("✅ Enhanced features provide significant value across all categories!")
    elif avg_improvement > 0:
        print("🟡 Enhanced features provide consistent modest improvements")
    else:
        print("⚠️  Enhanced features may not be adding significant value")
    
    print(f"\n🎯 MODELING STRATEGY CONFIRMED:")
    print(f"   • Train 6 category-level models (one per category)")
    print(f"   • Use same feature engineering for all (consistent patterns)")
    print(f"   • Features capture business patterns effectively")
    print(f"   • Scalable and maintainable approach")
    
    return all_models, all_results

def make_predictions(model, periods=365):
    """Make future predictions"""
    print(f"Making predictions for {periods} days...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Make predictions
    forecast = model.predict(future)
    
    print("Predictions completed!")
    return forecast

def plot_forecast_matplotlib(model, forecast, df):
    """Plot forecast using matplotlib"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main forecast plot
    model.plot(forecast, ax=axes[0,0])
    axes[0,0].set_title('Prophet Forecast')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Value')
    
    # Components plot
    model.plot_components(forecast, ax=axes[0,1])
    
    # Residuals
    residuals = df['y'] - forecast['yhat'][:len(df)]
    axes[1,0].plot(df['ds'], residuals)
    axes[1,0].set_title('Residuals')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Residual')
    
    # Distribution of residuals
    axes[1,1].hist(residuals.dropna(), bins=30, alpha=0.7)
    axes[1,1].set_title('Residuals Distribution')
    axes[1,1].set_xlabel('Residual')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_forecast_plotly(forecast, df, title="Prophet Forecast"):
    """Create interactive forecast plot using Plotly"""
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Actual',
        marker=dict(color='blue', size=3)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified'
    )
    
    fig.show()

def plot_components_plotly(forecast):
    """Plot Prophet components using Plotly"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Trend', 'Yearly Seasonality', 'Weekly Seasonality'],
        vertical_spacing=0.1
    )
    
    # Trend
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Yearly seasonality
    if 'yearly' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yearly'],
            mode='lines',
            name='Yearly',
            line=dict(color='green')
        ), row=2, col=1)
    
    # Weekly seasonality
    if 'weekly' in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['weekly'],
            mode='lines',
            name='Weekly',
            line=dict(color='orange')
        ), row=3, col=1)
    
    fig.update_layout(
        title='Prophet Forecast Components',
        height=800,
        showlegend=False
    )
    
    fig.show()

def evaluate_model(train_data, test_data, forecast, baselines):
    """Evaluate Prophet model performance against baselines"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Get predictions for test period
    test_start_idx = len(train_data)
    test_end_idx = test_start_idx + len(test_data)
    prophet_predictions = forecast['yhat'][test_start_idx:test_end_idx].values
    actual = test_data['y'].values
    
    # Calculate Prophet metrics
    mae_prophet = mean_absolute_error(actual, prophet_predictions)
    mse_prophet = mean_squared_error(actual, prophet_predictions)
    rmse_prophet = np.sqrt(mse_prophet)
    r2_prophet = r2_score(actual, prophet_predictions)
    
    print("\n=== PROPHET MODEL EVALUATION ===")
    print(f"Mean Absolute Error (MAE): {mae_prophet:.2f}")
    print(f"Mean Squared Error (MSE): {mse_prophet:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_prophet:.2f}")
    print(f"R² Score: {r2_prophet:.4f}")
    
    # Compare with baselines
    print("\n=== MODEL COMPARISON ===")
    print("Prophet vs Baselines (lower is better for MAE, MSE, RMSE; higher is better for R²):")
    
    for name, metrics in baselines.items():
        mae_improvement = ((metrics['MAE'] - mae_prophet) / metrics['MAE']) * 100
        rmse_improvement = ((metrics['RMSE'] - rmse_prophet) / metrics['RMSE']) * 100
        
        print(f"\nProphet vs {name.upper()}:")
        print(f"  MAE improvement: {mae_improvement:.1f}%")
        print(f"  RMSE improvement: {rmse_improvement:.1f}%")
        
        if mae_improvement > 0:
            print(f"  ✅ Prophet is better by {mae_improvement:.1f}%")
        else:
            print(f"  ❌ {name} baseline is better by {abs(mae_improvement):.1f}%")
    
    prophet_metrics = {'MAE': mae_prophet, 'MSE': mse_prophet, 'RMSE': rmse_prophet, 'R2': r2_prophet}
    return prophet_metrics

def create_baseline_predictions(train_data, test_data):
    """Create baseline predictions for comparison"""
    print("Creating baseline predictions...")
    
    # Baseline 1: Simple moving average (7-day window)
    window_size = 7
    moving_avg = train_data['y'].rolling(window=window_size).mean().iloc[-1]
    baseline_ma = [moving_avg] * len(test_data)
    
    # Baseline 2: Last value (naive forecast)
    last_value = train_data['y'].iloc[-1]
    baseline_naive = [last_value] * len(test_data)
    
    # Baseline 3: Seasonal naive (same day of week from previous week)
    baseline_seasonal = []
    for i in range(len(test_data)):
        if i < 7:
            # For first week, use last 7 days average
            baseline_seasonal.append(train_data['y'].iloc[-7:].mean())
        else:
            # Use value from same day of previous week
            baseline_seasonal.append(baseline_seasonal[i-7])
    
    baselines = {
        'moving_average': baseline_ma,
        'naive': baseline_naive,
        'seasonal_naive': baseline_seasonal
    }
    
    print("Baseline predictions created!")
    return baselines

def evaluate_baselines(test_data, baselines):
    """Evaluate baseline model performance"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    actual = test_data['y'].values
    
    print("\n=== Baseline Model Evaluation ===")
    baseline_metrics = {}
    
    for name, predictions in baselines.items():
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)
        
        baseline_metrics[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        
        print(f"\n{name.upper()} Baseline:")
        print(f"  MAE: {mae:.2f}")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
    
    return baseline_metrics

def optimize_product_parameters(product_data):
    """
    Simple parameter optimization for a single product using train/test split
    instead of cross-validation to avoid convergence issues
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import math
    
    # Split data into train/test (80/20)
    train_size = int(0.8 * len(product_data))
    train_data = product_data.iloc[:train_size]
    test_data = product_data.iloc[train_size:]
    
    param_grid = {
        'seasonality_mode': ['additive'],
        'changepoint_prior_scale': [0.05]
    }
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    # Generate parameter combinations
    param_combinations = []
    for seasonality in param_grid['seasonality_mode']:
        for changepoint in param_grid['changepoint_prior_scale']:
            param_combinations.append({
                'seasonality_mode': seasonality,
                'changepoint_prior_scale': changepoint
            })
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        try:
            print(f"--- Testing combination {i+1}/{len(param_combinations)} ---")
            print(f"Parameters: {params}")
            
            # Create and fit model
            model = Prophet(**params)
            model.fit(train_data)
            
            # Make predictions on test set
            future = model.make_future_dataframe(periods=len(test_data))
            forecast = model.predict(future)
            
            # Extract test predictions
            test_forecast = forecast.iloc[train_size:]['yhat']
            test_actual = test_data['y'].values
            
            # Calculate metrics
            rmse = math.sqrt(mean_squared_error(test_actual, test_forecast))
            mae = mean_absolute_error(test_actual, test_forecast)
            
            results.append({
                'params': params,
                'rmse': rmse,
                'mae': mae
            })
            
            print(f"RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            results.append({
                'params': params,
                'rmse': float('inf'),
                'mae': float('inf')
            })
            continue
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    return best_params, results
    """
    Perform cross-validation to find best Prophet parameters
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    import itertools
    
    print("Starting cross-validation for Prophet parameters...")
    
    best_params = None
    best_score = float('inf')
    cv_results = []
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\n--- Testing combination {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {param_dict}")
        
        try:
            # Create model with current parameters
            model = Prophet(
                yearly_seasonality=param_dict.get('yearly_seasonality', True),
                weekly_seasonality=param_dict.get('weekly_seasonality', True),
                daily_seasonality=param_dict.get('daily_seasonality', False),
                seasonality_mode=param_dict.get('seasonality_mode', 'additive'),
                changepoint_prior_scale=param_dict.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=param_dict.get('seasonality_prior_scale', 10.0)
            )
            
            # Fit model
            model.fit(df)
            
            # Perform cross-validation
            df_cv = cross_validation(
                model, 
                initial='730 days',  # 2 years initial training
                period='90 days',    # Step forward 90 days each fold
                horizon='90 days',   # Forecast 90 days ahead
                parallel="processes"
            )
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Use RMSE as the main metric
            avg_rmse = df_p['rmse'].mean()
            avg_mae = df_p['mae'].mean()
            
            # Handle MAPE carefully (might not exist for all data)
            avg_mape = df_p['mape'].mean() if 'mape' in df_p.columns else float('inf');
            
            cv_results.append({
                'params': param_dict,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'mape': avg_mape
            })
            
            print(f"Average RMSE: {avg_rmse:.4f}")
            
            # Track best parameters
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = param_dict
                
        except Exception as e:
            print(f"Error with parameters {param_dict}: {str(e)}")
            # Add failed result to maintain tracking
            cv_results.append({
                'params': param_dict,
                'rmse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf')
            })
            continue
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_score:.4f}")
    
    return best_params, cv_results

def cross_validate_by_product(df, n_products=5, param_grid=None):
    """
    Test if different products need different parameters
    """
    print("=== CROSS-VALIDATION BY PRODUCT ===")
    
    if param_grid is None:
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.1, 1.0, 10.0]
        }
    
    # Get unique products
    products = df['product_id'].unique()[:n_products]
    print(f"Testing {len(products)} products: {products}")
    
    product_results = {}
    
    for product in products:
        print(f"\n{'='*50}")
        print(f"Cross-validating for product: {product}")
        print(f"{'='*50}")
        
        # Filter data for this product
        product_data = df[df['product_id'] == product][['ds', 'y']].copy()
        product_data = product_data.sort_values('ds').reset_index(drop=True)
        
        if len(product_data) < 365:  # Need at least 1 year of data
            print(f"Skipping {product}: insufficient data ({len(product_data)} days)")
            continue
        
        # Run cross-validation for this product
        best_params, cv_results = cross_validate_prophet_parameters(
            product_data, param_grid
        )
        
        product_results[product] = {
            'best_params': best_params,
            'cv_results': cv_results,
            'data_points': len(product_data)
        }
    
    return product_results

def cross_validate_by_category(df, param_grid=None):
    """
    Test if different categories need different parameters
    """
    print("=== CROSS-VALIDATION BY CATEGORY ===")
    
    if param_grid is None:
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.1, 1.0, 10.0]
        }
    
    categories = df['category'].unique()
    print(f"Testing {len(categories)} categories: {categories}")
    
    category_results = {}
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Cross-validating for category: {category}")
        print(f"{'='*50}")
        
        # Aggregate data by category and date
        category_data = df[df['category'] == category].groupby('ds')['y'].sum().reset_index()
        category_data = category_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Category {category} has {len(category_data)} days of data")
        
        # Run cross-validation for this category
        best_params, cv_results = cross_validate_prophet_parameters(
            category_data, param_grid
        )
        
        category_results[category] = {
            'best_params': best_params,
            'cv_results': cv_results,
            'data_points': len(category_data)
        }
    
    return category_results

def analyze_parameter_variation(product_results, category_results):
    """
    Analyze if parameters vary significantly across products/categories
    """
    print("\n" + "="*60)
    print("PARAMETER VARIATION ANALYSIS")
    print("="*60)
    
    # Analyze product parameter variation
    if product_results:
        print("\n--- PRODUCT-LEVEL ANALYSIS ---")
        param_counts = {}
        
        for product, results in product_results.items():
            if results['best_params']:
                for param, value in results['best_params'].items():
                    if param not in param_counts:
                        param_counts[param] = {}
                    if value not in param_counts[param]:
                        param_counts[param][value] = 0
                    param_counts[param][value] += 1
        
        for param, counts in param_counts.items():
            total = sum(counts.values())
            print(f"\n{param}:")
            for value, count in counts.items():
                percentage = (count / total) * 100
                print(f"  {value}: {count}/{total} products ({percentage:.1f}%)")
            
            # Check if parameters are consistent
            most_common_count = max(counts.values())
            consistency = (most_common_count / total) * 100
            
            if consistency >= 80:
                print(f"  ✅ CONSISTENT: {consistency:.1f}% of products use the same value")
            else:
                print(f"  ⚠️  VARIABLE: Only {consistency:.1f}% of products use the most common value")
    
    # Analyze category parameter variation
    if category_results:
        print("\n--- CATEGORY-LEVEL ANALYSIS ---")
        param_counts = {}
        
        for category, results in category_results.items():
            if results['best_params']:
                for param, value in results['best_params'].items():
                    if param not in param_counts:
                        param_counts[param] = {}
                    if value not in param_counts[param]:
                        param_counts[param][value] = 0
                    param_counts[param][value] += 1
        
        for param, counts in param_counts.items():
            total = sum(counts.values())
            print(f"\n{param}:")
            for value, count in counts.items():
                percentage = (count / total) * 100
                print(f"  {value}: {count}/{total} categories ({percentage:.1f}%)")

def plot_cv_results(cv_results, title="Cross-Validation Results"):
    """
    Plot cross-validation results
    """
    if not cv_results:
        print("No CV results to plot")
        return
    
    import matplotlib.pyplot as plt
    
    # Extract metrics
    rmse_values = [r['rmse'] for r in cv_results]
    mae_values = [r['mae'] for r in cv_results]
    param_labels = [str(r['params']) for r in cv_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE plot
    ax1.bar(range(len(rmse_values)), rmse_values)
    ax1.set_title(f'{title} - RMSE')
    ax1.set_xlabel('Parameter Combination')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # MAE plot
    ax2.bar(range(len(mae_values)), mae_values)
    ax2.set_title(f'{title} - MAE')
    ax2.set_xlabel('Parameter Combination')
    ax2.set_ylabel('MAE')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main(run_cross_validation=True):
    """Main function to run the Prophet analysis"""
    # Load data
    df = load_and_prepare_data('daily_demand_by_product_modern.csv')
    
    # Use correct column names (ds and y already exist in your data)
    date_col = 'ds'
    value_col = 'y'
    
    print(f"Using columns: {date_col} (dates), {value_col} (values)")
    
    if run_cross_validation:
        print("\n" + "="*60)
        print("STARTING CROSS-VALIDATION ANALYSIS")
        print("="*60)
        
        # Define parameter grid for testing
        param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.1, 1.0, 10.0]
        }
        
        # Test parameters by product (sample of 3 products for speed)
        print("Testing parameter variation across products...")
        product_results = cross_validate_by_product(df, n_products=3, param_grid=param_grid)
        
        # Test parameters by category
        print("\nTesting parameter variation across categories...")
        category_results = cross_validate_by_category(df, param_grid=param_grid)
        
        # Analyze parameter variation
        analyze_parameter_variation(product_results, category_results)
        
        # Individual product detailed analysis
        print("\n" + "="*60)
        print("DETAILED INDIVIDUAL PRODUCT ANALYSIS")
        print("="*60)
        individual_results = analyze_individual_products(df, n_products=5)
        compare_product_characteristics(individual_results)
        plot_product_comparison(individual_results)
        
        # Plot results for first product/category
        if product_results:
            first_product = list(product_results.keys())[0]
            if product_results[first_product]['cv_results']:
                plot_cv_results(
                    product_results[first_product]['cv_results'], 
                    f"CV Results - {first_product}"
                )
        
        # Recommendation based on analysis
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("Based on cross-validation results:")
        print("1. If parameters are consistent (>80%) across products/categories:")
        print("   → Use a UNIVERSAL model with the most common parameters")
        print("2. If parameters vary significantly (<80% consistency):")
        print("   → Use INDIVIDUAL models for each product/category")
        print("3. Consider the trade-off between accuracy and complexity")
        
        return product_results, category_results
    
    else:
        # Original single-model approach
        # Aggregate all products by date
        prophet_data = df.groupby('ds')['y'].sum().reset_index()
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        # Split into train/test
        split_index = int(len(prophet_data) * 0.8)
        train_data = prophet_data[:split_index].copy()
        test_data = prophet_data[split_index:].copy()
        
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Create and fit model
        model = create_prophet_model(train_data)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        
        # Create baselines
        baselines = create_baseline_predictions(train_data, test_data)
        baseline_metrics = evaluate_baselines(test_data, baselines)
        
        # Evaluate model
        metrics = evaluate_model(train_data, test_data, forecast, baseline_metrics)
        
        # Visualizations
        print("\nCreating visualizations...")
        plot_forecast_matplotlib(model, forecast, prophet_data)
        plot_forecast_plotly(forecast, prophet_data)
        plot_components_plotly(forecast)
        
        return model, forecast, metrics

def analyze_individual_products(df, n_products=5):
    """
    Analyze individual products in detail with simpler parameter grid
    """
    print("="*60)
    print("INDIVIDUAL PRODUCT ANALYSIS")
    print("="*60)
    
    # Simpler parameter grid for products (to avoid convergence issues)
    param_grid = {
        'seasonality_mode': ['additive'],  # Only additive to reduce errors
        'changepoint_prior_scale': [0.05]  # Single conservative value
    }
    
    # Get products with most data
    product_counts = df['product_id'].value_counts()
    top_products = product_counts.head(n_products).index.tolist()
    
    print(f"Analyzing top {n_products} products by data volume:")
    for i, product in enumerate(top_products, 1):
        count = product_counts[product]
        print(f"  {i}. {product}: {count} data points")
    
    product_results = {}
    
    for product in top_products:
        print(f"\n{'='*60}")
        print(f"ANALYZING PRODUCT: {product}")
        print(f"{'='*60}")
        
        # Filter data for this product
        product_data = df[df['product_id'] == product][['ds', 'y']].copy()
        product_data = product_data.sort_values('ds').reset_index(drop=True)
        
        print(f"Data points: {len(product_data)}")
        print(f"Date range: {product_data['ds'].min()} to {product_data['ds'].max()}")
        print(f"Value range: {product_data['y'].min():.2f} to {product_data['y'].max():.2f}")
        print(f"Average daily demand: {product_data['y'].mean():.2f}")
        
        # Check for sufficient data
        if len(product_data) < 730:  # Need at least 2 years
            print(f"⚠️  Insufficient data for robust cross-validation ({len(product_data)} days)")
            continue
        
        # Analyze data characteristics
        print("\nData Characteristics:")
        print(f"  Standard deviation: {product_data['y'].std():.2f}")
        print(f"  Coefficient of variation: {(product_data['y'].std() / product_data['y'].mean() * 100):.1f}%")
        
        # Check for zeros or negative values
        zero_count = (product_data['y'] == 0).sum()
        negative_count = (product_data['y'] < 0).sum()
        if zero_count > 0:
            print(f"  Zero values: {zero_count} ({zero_count/len(product_data)*100:.1f}%)")
        if negative_count > 0:
            print(f"  Negative values: {negative_count}")
        
        # Try simple Prophet model first
        try:
            print("\nTesting basic Prophet model...")
            basic_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            basic_model.fit(product_data)
            
            # Simple forecast
            future = basic_model.make_future_dataframe(periods=30)
            forecast = basic_model.predict(future)
            
            # Calculate simple RMSE on training data
            from sklearn.metrics import mean_squared_error
            train_rmse = np.sqrt(mean_squared_error(
                product_data['y'], 
                forecast['yhat'][:len(product_data)]
            ))
            print(f"  Basic model RMSE: {train_rmse:.2f}")
            
            # Now try parameter optimization
            print("Optimizing parameters...")
            best_params, cv_results = optimize_product_parameters(product_data)
            
            product_results[product] = {
                'best_params': best_params,
                'cv_results': cv_results,
                'data_points': len(product_data),
                'basic_rmse': train_rmse,
                'avg_demand': product_data['y'].mean(),
                'std_demand': product_data['y'].std()
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {product}: {str(e)}")
            product_results[product] = {
                'error': str(e),
                'data_points': len(product_data),
                'avg_demand': product_data['y'].mean(),
                'std_demand': product_data['y'].std()
            }
    
    return product_results

def compare_product_characteristics(product_results):
    """
    Compare characteristics across products
    """
    print("\n" + "="*60)
    print("PRODUCT COMPARISON SUMMARY")
    print("="*60)
    
    successful_products = {k: v for k, v in product_results.items() if 'best_params' in v}
    failed_products = {k: v for k, v in product_results.items() if 'error' in v}
    
    print(f"Successfully analyzed: {len(successful_products)} products")
    print(f"Failed to analyze: {len(failed_products)} products")
    
    if successful_products:
        print("\n--- SUCCESSFUL PRODUCTS ---")
        for product, results in successful_products.items():
            best_rmse = min([r['rmse'] for r in results['cv_results']]) if results['cv_results'] else 'N/A'
            print(f"{product}:")
            print(f"  Best RMSE: {best_rmse}")
            print(f"  Best params: {results['best_params']}")
            print(f"  Avg demand: {results['avg_demand']:.2f}")
            print(f"  Demand volatility: {results['std_demand']:.2f}")
    
    if failed_products:
        print("\n--- FAILED PRODUCTS ---")
        for product, results in failed_products.items():
            print(f"{product}:")
            print(f"  Error: {results['error']}")
            print(f"  Avg demand: {results['avg_demand']:.2f}")
            print(f"  Data points: {results['data_points']}")
    
    # Parameter analysis for successful products
    if len(successful_products) > 1:
        print("\n--- PARAMETER VARIATION ANALYSIS ---")
        param_counts = {}
        
        for product, results in successful_products.items():
            if results['best_params']:
                for param, value in results['best_params'].items():
                    if param not in param_counts:
                        param_counts[param] = {}
                    if value not in param_counts[param]:
                        param_counts[param][value] = 0
                    param_counts[param][value] += 1
        
        for param, counts in param_counts.items():
            total = sum(counts.values())
            print(f"\n{param}:")
            for value, count in counts.items():
                percentage = (count / total) * 100
                print(f"  {value}: {count}/{total} products ({percentage:.1f}%)")
            
            # Check consistency
            most_common_count = max(counts.values())
            consistency = (most_common_count / total) * 100
            
            if consistency >= 80:
                print(f"  ✅ CONSISTENT: {consistency:.1f}% of products use the same value")
            else:
                print(f"  ⚠️  VARIABLE: Only {consistency:.1f}% of products use the most common value")

def plot_product_comparison(product_results):
    """
    Create visualizations comparing products
    """
    successful_products = {k: v for k, v in product_results.items() if 'best_params' in v}
    
    if len(successful_products) < 2:
        print("Not enough successful products for comparison plots")
        return
    
    import matplotlib.pyplot as plt
    
    # Prepare data for plotting
    products = list(successful_products.keys())
    avg_demands = [results['avg_demand'] for results in successful_products.values()]
    std_demands = [results['std_demand'] for results in successful_products.values()]
    best_rmses = []
    
    for results in successful_products.values():
        if results['cv_results']:
            best_rmse = min([r['rmse'] for r in results['cv_results']])
            best_rmses.append(best_rmse)
        else:
            best_rmses.append(results.get('basic_rmse', 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average demand by product
    axes[0,0].bar(range(len(products)), avg_demands)
    axes[0,0].set_title('Average Daily Demand by Product')
    axes[0,0].set_xlabel('Product')
    axes[0,0].set_ylabel('Average Demand')
    axes[0,0].set_xticks(range(len(products)))
    axes[0,0].set_xticklabels([p.split('_')[-1] for p in products], rotation=45)
    
    # Plot 2: Demand volatility (std dev)
    axes[0,1].bar(range(len(products)), std_demands)
    axes[0,1].set_title('Demand Volatility by Product')
    axes[0,1].set_xlabel('Product')
    axes[0,1].set_ylabel('Standard Deviation')
    axes[0,1].set_xticks(range(len(products)))
    axes[0,1].set_xticklabels([p.split('_')[-1] for p in products], rotation=45)
    
    # Plot 3: Best RMSE by product
    axes[1,0].bar(range(len(products)), best_rmses)
    axes[1,0].set_title('Best Cross-Validation RMSE by Product')
    axes[1,0].set_xlabel('Product')
    axes[1,0].set_ylabel('RMSE')
    axes[1,0].set_xticks(range(len(products)))
    axes[1,0].set_xticklabels([p.split('_')[-1] for p in products], rotation=45)
    
    # Plot 4: Demand vs RMSE scatter
    axes[1,1].scatter(avg_demands, best_rmses)
    axes[1,1].set_title('Demand Level vs Forecast Accuracy')
    axes[1,1].set_xlabel('Average Daily Demand')
    axes[1,1].set_ylabel('Best RMSE')
    
    # Add product labels to scatter plot
    for i, product in enumerate(products):
        axes[1,1].annotate(product.split('_')[-1], 
                          (avg_demands[i], best_rmses[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def analyze_seasonality_patterns(df, n_products=10):
    """
    Comprehensive analysis to detect different seasonality patterns across products
    """
    print("="*80)
    print("SEASONALITY PATTERN ANALYSIS")
    print("="*80)
    
    from scipy import stats
    from scipy.fft import fft, fftfreq
    import pandas as pd
    import numpy as np
    
    # Get products with most data for reliable seasonality analysis
    product_counts = df['product_id'].value_counts()
    top_products = product_counts.head(n_products).index.tolist()
    
    seasonality_results = {}
    
    print(f"Analyzing seasonality patterns for top {n_products} products...")
    
    for product in top_products:
        print(f"\n{'='*50}")
        print(f"ANALYZING SEASONALITY: {product}")
        print(f"{'='*50}")
        
        # Filter data for this product
        product_data = df[df['product_id'] == product][['ds', 'y']].copy()
        product_data = product_data.sort_values('ds').reset_index(drop=True)
        product_data['ds'] = pd.to_datetime(product_data['ds'])
        
        if len(product_data) < 365:
            print(f"⚠️  Insufficient data for seasonality analysis ({len(product_data)} days)")
            continue
        
        # Extract time features
        product_data['month'] = product_data['ds'].dt.month
        product_data['quarter'] = product_data['ds'].dt.quarter
        product_data['day_of_week'] = product_data['ds'].dt.dayofweek
        product_data['day_of_year'] = product_data['ds'].dt.dayofyear
        
        seasonality_analysis = {}
        
        # 1. YEARLY SEASONALITY ANALYSIS
        print("\n--- YEARLY SEASONALITY ---")
        monthly_avg = product_data.groupby('month')['y'].mean()
        quarterly_avg = product_data.groupby('quarter')['y'].mean()
        
        # Calculate coefficient of variation for months
        monthly_cv = monthly_avg.std() / monthly_avg.mean()
        quarterly_cv = quarterly_avg.std() / quarterly_avg.mean()
        
        print(f"Monthly coefficient of variation: {monthly_cv:.3f}")
        print(f"Quarterly coefficient of variation: {quarterly_cv:.3f}")
        
        # Statistical test for monthly differences
        month_groups = [product_data[product_data['month'] == m]['y'].values for m in range(1, 13)]
        month_groups = [group for group in month_groups if len(group) > 0]
        
        if len(month_groups) >= 3:
            yearly_f_stat, yearly_p_value = stats.f_oneway(*month_groups)
            yearly_significant = yearly_p_value < 0.05
            print(f"ANOVA F-statistic: {yearly_f_stat:.3f}, p-value: {yearly_p_value:.4f}")
            print(f"Yearly seasonality: {'SIGNIFICANT' if yearly_significant else 'NOT SIGNIFICANT'}")
        else:
            yearly_significant = False
            yearly_p_value = 1.0
        
        # Find peak and low months
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        peak_ratio = monthly_avg.max() / monthly_avg.min()
        
        print(f"Peak month: {peak_month} (avg: {monthly_avg[peak_month]:.2f})")
        print(f"Low month: {low_month} (avg: {monthly_avg[low_month]:.2f})")
        print(f"Peak/Low ratio: {peak_ratio:.2f}")
        
        seasonality_analysis['yearly'] = {
            'significant': yearly_significant,
            'p_value': yearly_p_value,
            'monthly_cv': monthly_cv,
            'peak_month': peak_month,
            'low_month': low_month,
            'peak_ratio': peak_ratio,
            'monthly_avg': monthly_avg.to_dict()
        }
        
        # 2. WEEKLY SEASONALITY ANALYSIS
        print("\n--- WEEKLY SEASONALITY ---")
        weekly_avg = product_data.groupby('day_of_week')['y'].mean()
        weekly_cv = weekly_avg.std() / weekly_avg.mean()
        
        print(f"Weekly coefficient of variation: {weekly_cv:.3f}")
        
        # Statistical test for weekly differences
        dow_groups = [product_data[product_data['day_of_week'] == d]['y'].values for d in range(7)]
        dow_groups = [group for group in dow_groups if len(group) > 0]
        
        if len(dow_groups) >= 3:
            weekly_f_stat, weekly_p_value = stats.f_oneway(*dow_groups)
            weekly_significant = weekly_p_value < 0.05
            print(f"ANOVA F-statistic: {weekly_f_stat:.3f}, p-value: {weekly_p_value:.4f}")
            print(f"Weekly seasonality: {'SIGNIFICANT' if weekly_significant else 'NOT SIGNIFICANT'}")
        else:
            weekly_significant = False
            weekly_p_value = 1.0
        
        # Find peak and low days
        peak_day = weekly_avg.idxmax()
        low_day = weekly_avg.idxmin()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        print(f"Peak day: {day_names[peak_day]} (avg: {weekly_avg[peak_day]:.2f})")
        print(f"Low day: {day_names[low_day]} (avg: {weekly_avg[low_day]:.2f})")
        
        seasonality_analysis['weekly'] = {
            'significant': weekly_significant,
            'p_value': weekly_p_value,
            'weekly_cv': weekly_cv,
            'peak_day': peak_day,
            'low_day': low_day,
            'weekly_avg': weekly_avg.to_dict()
        }
        
        # 3. FOURIER ANALYSIS FOR HIDDEN CYCLES
        print("\n--- FREQUENCY ANALYSIS ---")
        if len(product_data) >= 730:  # Need at least 2 years for reliable frequency analysis
            y_values = product_data['y'].values
            # Remove trend using first difference
            y_detrended = np.diff(y_values)
            
            # Apply FFT
            fft_values = fft(y_detrended)
            frequencies = fftfreq(len(y_detrended), d=1)  # Daily frequency
            
            # Find dominant frequencies
            power = np.abs(fft_values)**2
            # Only look at positive frequencies
            pos_mask = frequencies > 0
            pos_freqs = frequencies[pos_mask]
            pos_power = power[pos_mask]
            
            # Find top 3 frequencies
            top_indices = np.argsort(pos_power)[-3:][::-1]
            dominant_freqs = pos_freqs[top_indices]
            dominant_periods = 1 / dominant_freqs  # Convert to periods in days
            
            print("Top 3 dominant cycles:")
            for i, period in enumerate(dominant_periods):
                if period >= 300:  # Yearly cycles
                    cycle_type = f"~{period/365:.1f} years"
                elif period >= 25:  # Monthly cycles
                    cycle_type = f"~{period/30:.1f} months"
                elif period >= 6:  # Weekly cycles
                    cycle_type = f"~{period/7:.1f} weeks"
                else:
                    cycle_type = f"{period:.1f} days"
                print(f"  {i+1}. {period:.1f} days ({cycle_type})")
            
            seasonality_analysis['fourier'] = {
                'dominant_periods': dominant_periods.tolist(),
                'dominant_powers': pos_power[top_indices].tolist()
            }
        
        # 4. CATEGORY INFORMATION
        category = df[df['product_id'] == product]['category'].iloc[0]
        seasonality_analysis['category'] = category
        seasonality_analysis['data_points'] = len(product_data)
        
        seasonality_results[product] = seasonality_analysis
    
    return seasonality_results

def compare_seasonality_across_products(seasonality_results):
    """
    Compare seasonality patterns across products to identify clusters
    """
    print("\n" + "="*80)
    print("SEASONALITY COMPARISON ACROSS PRODUCTS")
    print("="*80)
    
    # Analyze yearly seasonality patterns
    print("\n--- YEARLY SEASONALITY COMPARISON ---")
    yearly_significant_products = []
    yearly_patterns = {}
    
    for product, analysis in seasonality_results.items():
        if 'yearly' in analysis and analysis['yearly']['significant']:
            yearly_significant_products.append(product)
            peak_month = analysis['yearly']['peak_month']
            if peak_month not in yearly_patterns:
                yearly_patterns[peak_month] = []
            yearly_patterns[peak_month].append(product)
    
    print(f"Products with significant yearly seasonality: {len(yearly_significant_products)}/{len(seasonality_results)}")
    
    if yearly_patterns:
        print("\nYearly seasonality patterns by peak month:")
        for month, products in sorted(yearly_patterns.items()):
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            print(f"  {month_names[month]} peak: {len(products)} products")
            for product in products:
                category = seasonality_results[product]['category']
                ratio = seasonality_results[product]['yearly']['peak_ratio']
                print(f"    - {product} ({category}) - Peak/Low ratio: {ratio:.2f}")
    
    # Analyze weekly seasonality patterns
    print("\n--- WEEKLY SEASONALITY COMPARISON ---")
    weekly_significant_products = []
    weekly_patterns = {}
    
    for product, analysis in seasonality_results.items():
        if 'weekly' in analysis and analysis['weekly']['significant']:
            weekly_significant_products.append(product)
            peak_day = analysis['weekly']['peak_day']
            if peak_day not in weekly_patterns:
                weekly_patterns[peak_day] = []
            weekly_patterns[peak_day].append(product)
    
    print(f"Products with significant weekly seasonality: {len(weekly_significant_products)}/{len(seasonality_results)}")
    
    if weekly_patterns:
        print("\nWeekly seasonality patterns by peak day:")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day, products in sorted(weekly_patterns.items()):
            print(f"  {day_names[day]} peak: {len(products)} products")
            for product in products:
                category = seasonality_results[product]['category']
                print(f"    - {product} ({category})")
    
    # Category-based seasonality analysis
    print("\n--- SEASONALITY BY CATEGORY ---")
    category_seasonality = {}
    
    for product, analysis in seasonality_results.items():
        category = analysis['category']
        if category not in category_seasonality:
            category_seasonality[category] = {
                'yearly_significant': 0,
                'weekly_significant': 0,
                'total_products': 0,
                'peak_months': [],
                'peak_days': []
            }
        
        category_seasonality[category]['total_products'] += 1
        
        if analysis.get('yearly', {}).get('significant', False):
            category_seasonality[category]['yearly_significant'] += 1
            category_seasonality[category]['peak_months'].append(
                analysis['yearly']['peak_month']
            )
        
        if analysis.get('weekly', {}).get('significant', False):
            category_seasonality[category]['weekly_significant'] += 1
            category_seasonality[category]['peak_days'].append(
                analysis['weekly']['peak_day']
            )
    
    for category, stats in category_seasonality.items():
        total = stats['total_products']
        yearly_pct = (stats['yearly_significant'] / total) * 100
        weekly_pct = (stats['weekly_significant'] / total) * 100
        
        print(f"\n{category.upper()}:")
        print(f"  Products analyzed: {total}")
        print(f"  Yearly seasonality: {stats['yearly_significant']}/{total} ({yearly_pct:.1f}%)")
        print(f"  Weekly seasonality: {stats['weekly_significant']}/{total} ({weekly_pct:.1f}%)")
        
        # Most common peak months/days for this category
        if stats['peak_months']:
            from collections import Counter
            common_months = Counter(stats['peak_months']).most_common(2)
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            print(f"  Common peak months: {[f'{month_names[m]}({c})' for m, c in common_months]}")
        
        if stats['peak_days']:
            from collections import Counter
            common_days = Counter(stats['peak_days']).most_common(2)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            print(f"  Common peak days: {[f'{day_names[d]}({c})' for d, c in common_days]}")
    
    return category_seasonality

def prophet_seasonality_recommendations(seasonality_results, category_seasonality):
    """
    Provide recommendations for Prophet modeling based on seasonality analysis
    """
    print("\n" + "="*80)
    print("PROPHET MODELING RECOMMENDATIONS")
    print("="*80)
    
    total_products = len(seasonality_results)
    yearly_significant = sum(1 for r in seasonality_results.values() 
                           if r.get('yearly', {}).get('significant', False))
    weekly_significant = sum(1 for r in seasonality_results.values() 
                           if r.get('weekly', {}).get('significant', False))
    
    yearly_pct = (yearly_significant / total_products) * 100
    weekly_pct = (weekly_significant / total_products) * 100
    
    print(f"OVERALL SEASONALITY SUMMARY:")
    print(f"  Products with yearly seasonality: {yearly_significant}/{total_products} ({yearly_pct:.1f}%)")
    print(f"  Products with weekly seasonality: {weekly_significant}/{total_products} ({weekly_pct:.1f}%)")
    
    print(f"\n🔍 PROPHET CAN HANDLE YOUR SEASONALITY PATTERNS:")
    print(f"  ✅ Prophet supports yearly seasonality (Fourier series)")
    print(f"  ✅ Prophet supports weekly seasonality (Fourier series)")
    print(f"  ✅ Prophet supports custom seasonalities")
    print(f"  ✅ Prophet can model additive or multiplicative seasonality")
    
    print(f"\n📊 MODELING STRATEGY RECOMMENDATIONS:")
    
    # Check if categories have consistent seasonality
    consistent_categories = []
    variable_categories = []
    
    for category, stats in category_seasonality.items():
        yearly_consistency = (stats['yearly_significant'] / stats['total_products']) * 100
        if yearly_consistency >= 80 or yearly_consistency <= 20:
            consistent_categories.append(category)
        else:
            variable_categories.append(category)
    
    if len(consistent_categories) == len(category_seasonality):
        print(f"\n🎯 RECOMMENDATION: CATEGORY-LEVEL MODELS")
        print(f"   → Use {len(category_seasonality)} Prophet models (one per category)")
        print(f"   → Categories show consistent seasonality patterns within each group")
        print(f"   → This reduces complexity from 2000 to {len(category_seasonality)} models")
        
        print(f"\n   Category-specific settings:")
        for category, stats in category_seasonality.items():
            yearly_pct = (stats['yearly_significant'] / stats['total_products']) * 100
            weekly_pct = (stats['weekly_significant'] / stats['total_products']) * 100
            
            yearly_setting = yearly_pct >= 50
            weekly_setting = weekly_pct >= 50
            
            print(f"   • {category}:")
            print(f"     - yearly_seasonality={yearly_setting}")
            print(f"     - weekly_seasonality={weekly_setting}")
            
            if stats['peak_months']:
                from collections import Counter
                most_common_month = Counter(stats['peak_months']).most_common(1)[0][0]
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                print(f"     - Peak season: {month_names[most_common_month]}")
    
    else:
        print(f"\n⚠️  RECOMMENDATION: HYBRID APPROACH")
        print(f"   → Some categories have variable seasonality patterns")
        print(f"   → Consider individual models for products in variable categories")
        print(f"   → Use category models for consistent categories")
        
        print(f"\n   Consistent categories (use category-level models):")
        for cat in consistent_categories:
            print(f"     • {cat}")
        
        print(f"\n   Variable categories (consider individual models):")
        for cat in variable_categories:
            print(f"     • {cat}")
    
    print(f"\n🚀 PROPHET vs OTHER MODELS:")
    print(f"   ✅ STICK WITH PROPHET if:")
    print(f"      - You want automatic seasonality detection")
    print(f"      - You need uncertainty intervals")
    print(f"      - You want holiday effects modeling")
    print(f"      - You need interpretable components")
    
    print(f"\n   🤔 CONSIDER ALTERNATIVES if:")
    print(f"      - You have very complex, irregular seasonalities")
    print(f"      - You need real-time, high-frequency predictions")
    print(f"      - You have strong feature interactions")
    
    print(f"\n   📚 ALTERNATIVE MODELS:")
    print(f"      - SARIMA: For classical time series with known seasonality")
    print(f"      - XGBoost/Random Forest: For feature-rich modeling")
    print(f"      - LSTM/Neural Networks: For complex patterns")
    print(f"      - Seasonal decomposition + ML: Hybrid approach")

def comprehensive_seasonality_detection(df, n_products=10):
    """
    Comprehensive seasonality detection WITHOUT using Prophet
    Uses statistical methods, decomposition, and spectral analysis
    """
    print("="*80)
    print("COMPREHENSIVE SEASONALITY DETECTION (NO PROPHET)")
    print("="*80)
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.fft import fft, fftfreq
    from scipy.signal import periodogram
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get products with most data
    product_counts = df['product_id'].value_counts()
    top_products = product_counts.head(n_products).index.tolist()
    
    seasonality_analysis = {}
    
    print(f"Analyzing {n_products} products for seasonality patterns...")
    
    for i, product in enumerate(top_products, 1):
        print(f"\n{'='*70}")
        print(f"PRODUCT {i}/{n_products}: {product}")
        print(f"{'='*70}")
        
        # Filter data for this product
        product_data = df[df['product_id'] == product][['ds', 'y']].copy()
        product_data = product_data.sort_values('ds').reset_index(drop=True)
        product_data['ds'] = pd.to_datetime(product_data['ds'])
        product_data.set_index('ds', inplace=True)
        
        category = df[df['product_id'] == product]['category'].iloc[0]
        print(f"Category: {category}")
        print(f"Data points: {len(product_data)}")
        print(f"Date range: {product_data.index.min()} to {product_data.index.max()}")
        
        if len(product_data) < 365:
            print("⚠️  Insufficient data for reliable seasonality analysis")
            continue
        
        # Initialize results for this product
        product_analysis = {
            'product_id': product,
            'category': category,
            'data_points': len(product_data),
            'tests': {}
        }
        
        # =============================================================
        # 1. BASIC STATISTICS AND STATIONARITY
        # =============================================================
        print("\n--- 1. BASIC STATISTICS ---")
        y = product_data['y'].values
        mean_val = np.mean(y)
        std_val = np.std(y)
        cv = std_val / mean_val if mean_val != 0 else 0
        
        print(f"Mean: {mean_val:.2f}")
        print(f"Std Dev: {std_val:.2f}")
        print(f"Coefficient of Variation: {cv:.3f}")
        
        # Test for stationarity
        adf_result = adfuller(y.dropna())
        is_stationary = adf_result[1] < 0.05
        print(f"Stationarity (ADF test): {'STATIONARY' if is_stationary else 'NON-STATIONARY'} (p={adf_result[1]:.4f})")
        
        product_analysis['basic_stats'] = {
            'mean': mean_val,
            'std': std_val,
            'cv': cv,
            'is_stationary': is_stationary,
            'adf_pvalue': adf_result[1]
        }
        
        # =============================================================
        # 2. SEASONAL DECOMPOSITION
        # =============================================================
        print("\n--- 2. SEASONAL DECOMPOSITION ---")
        try:
            # Try different periods for decomposition
            decomposition_results = {}
            
            # Weekly decomposition (if enough data)
            if len(product_data) >= 14:
                weekly_decomp = seasonal_decompose(product_data['y'], model='additive', period=7, extrapolate_trend='freq')
                weekly_seasonal_strength = np.var(weekly_decomp.seasonal) / np.var(weekly_decomp.resid + weekly_decomp.seasonal)
                decomposition_results['weekly'] = weekly_seasonal_strength
                print(f"Weekly seasonal strength: {weekly_seasonal_strength:.4f}")
            
            # Monthly decomposition (if enough data)
            if len(product_data) >= 60:
                monthly_decomp = seasonal_decompose(product_data['y'], model='additive', period=30, extrapolate_trend='freq')
                monthly_seasonal_strength = np.var(monthly_decomp.seasonal) / np.var(monthly_decomp.resid + monthly_decomp.seasonal)
                decomposition_results['monthly'] = monthly_seasonal_strength
                print(f"Monthly seasonal strength: {monthly_seasonal_strength:.4f}")
            
            # Yearly decomposition (if enough data)
            if len(product_data) >= 730:
                yearly_decomp = seasonal_decompose(product_data['y'], model='additive', period=365, extrapolate_trend='freq')
                yearly_seasonal_strength = np.var(yearly_decomp.seasonal) / np.var(yearly_decomp.resid + yearly_decomp.seasonal)
                decomposition_results['yearly'] = yearly_seasonal_strength
                print(f"Yearly seasonal strength: {yearly_seasonal_strength:.4f}")
            
            product_analysis['decomposition'] = decomposition_results
            
        except Exception as e:
            print(f"Decomposition failed: {e}")
            product_analysis['decomposition'] = {}
        
        # =============================================================
        # 3. SPECTRAL ANALYSIS (PERIODOGRAM)
        # =============================================================
        print("\n--- 3. SPECTRAL ANALYSIS ---")
        try:
            # Remove trend first
            y_detrended = np.diff(y)
            
            # Periodogram to find dominant frequencies
            frequencies, power = periodogram(y_detrended, fs=1.0)  # fs=1 for daily data
            
            # Convert frequencies to periods (in days)
            periods = 1 / frequencies[1:]  # Skip DC component
            power = power[1:]
            
            # Find top 5 dominant periods
            top_indices = np.argsort(power)[-5:][::-1]
            dominant_periods = periods[top_indices]
            dominant_powers = power[top_indices]
            
            print("Top 5 dominant cycles:")
            spectral_results = []
            for j, (period, power_val) in enumerate(zip(dominant_periods, dominant_powers)):
                if period >= 350:  # Yearly
                    cycle_type = f"Yearly (~{period/365:.1f} years)"
                elif 25 <= period <= 35:  # Monthly
                    cycle_type = f"Monthly (~{period/30:.1f} months)"
                elif 6 <= period <= 8:  # Weekly
                    cycle_type = "Weekly"
                elif period < 6:
                    cycle_type = f"Short ({period:.1f} days)"
                else:
                    cycle_type = f"Other ({period:.1f} days)"
                
                print(f"  {j+1}. {period:.1f} days - {cycle_type} (power: {power_val:.2e})")
                spectral_results.append({
                    'period': period,
                    'power': power_val,
                    'type': cycle_type
                })
            
            product_analysis['spectral'] = spectral_results
            
        except Exception as e:
            print(f"Spectral analysis failed: {e}")
            product_analysis['spectral'] = []
        
        # =============================================================
        # 4. STATISTICAL TESTS FOR SEASONALITY
        # =============================================================
        print("\n--- 4. STATISTICAL SEASONALITY TESTS ---")
        
        # Add time features
        product_data['month'] = product_data.index.month
        product_data['quarter'] = product_data.index.quarter
        product_data['day_of_week'] = product_data.index.dayofweek
        product_data['day_of_year'] = product_data.index.dayofyear
        
        # Test yearly seasonality (monthly differences)
        month_groups = [product_data[product_data['month'] == m]['y'].values for m in range(1, 13)]
        month_groups = [group for group in month_groups if len(group) > 0]
        
        if len(month_groups) >= 3:
            yearly_f_stat, yearly_p = stats.f_oneway(*month_groups)
            yearly_significant = yearly_p < 0.05
            print(f"Yearly seasonality (ANOVA): F={yearly_f_stat:.3f}, p={yearly_p:.4f} - {'SIGNIFICANT' if yearly_significant else 'NOT SIGNIFICANT'}")
            
            # Calculate effect size (eta-squared)
            yearly_eta_squared = yearly_f_stat * (len(month_groups) - 1) / (yearly_f_stat * (len(month_groups) - 1) + len(product_data) - len(month_groups))
            print(f"Effect size (eta²): {yearly_eta_squared:.4f}")
        else:
            yearly_significant = False
            yearly_p = 1.0
            yearly_eta_squared = 0
        
        # Test weekly seasonality (day of week differences)
        dow_groups = [product_data[product_data['day_of_week'] == d]['y'].values for d in range(7)]
        dow_groups = [group for group in dow_groups if len(group) > 0]
        
        if len(dow_groups) >= 3:
            weekly_f_stat, weekly_p = stats.f_oneway(*dow_groups)
            weekly_significant = weekly_p < 0.05
            print(f"Weekly seasonality (ANOVA): F={weekly_f_stat:.3f}, p={weekly_p:.4f} - {'SIGNIFICANT' if weekly_significant else 'NOT SIGNIFICANT'}")
            
            # Calculate effect size
            weekly_eta_squared = weekly_f_stat * (len(dow_groups) - 1) / (weekly_f_stat * (len(dow_groups) - 1) + len(product_data) - len(dow_groups))
            print(f"Effect size (eta²): {weekly_eta_squared:.4f}")
        else:
            weekly_significant = False
            weekly_p = 1.0
            weekly_eta_squared = 0
        
        # Kruskal-Wallis test (non-parametric alternative)
        if len(month_groups) >= 3:
            yearly_kw_stat, yearly_kw_p = stats.kruskal(*month_groups)
            print(f"Yearly seasonality (Kruskal-Wallis): H={yearly_kw_stat:.3f}, p={yearly_kw_p:.4f}")
        
        if len(dow_groups) >= 3:
            weekly_kw_stat, weekly_kw_p = stats.kruskal(*dow_groups)
            print(f"Weekly seasonality (Kruskal-Wallis): H={weekly_kw_stat:.3f}, p={weekly_kw_p:.4f}")
        
        product_analysis['statistical_tests'] = {
            'yearly': {
                'significant': yearly_significant,
                'p_value': yearly_p,
                'effect_size': yearly_eta_squared,
                'test_type': 'ANOVA'
            },
            'weekly': {
                'significant': weekly_significant,
                'p_value': weekly_p,
                'effect_size': weekly_eta_squared,
                'test_type': 'ANOVA'
            }
        }
        
        # =============================================================
        # 5. AUTOCORRELATION ANALYSIS
        # =============================================================
        print("\n--- 5. AUTOCORRELATION ANALYSIS ---")
        
        from statsmodels.tsa.stattools import acf, pacf
        
        # Calculate autocorrelation
        max_lags = min(100, len(product_data) // 4)
        autocorr = acf(product_data['y'], nlags=max_lags, fft=True)
        
        # Look for significant autocorrelations at seasonal lags
        seasonal_lags = [7, 14, 21, 28, 30, 60, 90, 365]  # Weekly, monthly, yearly
        seasonal_autocorr = {}
        
        for lag in seasonal_lags:
            if lag < len(autocorr):
                seasonal_autocorr[lag] = autocorr[lag]
                lag_type = "Weekly" if lag <= 28 else "Monthly" if lag <= 90 else "Yearly"
                print(f"Autocorr at lag {lag} ({lag_type}): {autocorr[lag]:.4f}")
        
        product_analysis['autocorrelation'] = seasonal_autocorr
        
        # =============================================================
        # 6. TREND ANALYSIS
        # =============================================================
        print("\n--- 6. TREND ANALYSIS ---")
        
        # Linear trend
        x = np.arange(len(product_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, product_data['y'])
        
        trend_direction = "INCREASING" if slope > 0 else "DECREASING" if slope < 0 else "FLAT"
        trend_significant = p_value < 0.05
        
        print(f"Linear trend: {trend_direction} (slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f})")
        print(f"Trend significance: {'SIGNIFICANT' if trend_significant else 'NOT SIGNIFICANT'}")
        
        product_analysis['trend'] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': trend_significant,
            'direction': trend_direction
        }
        
        # =============================================================
        # 7. SEASONALITY STRENGTH SUMMARY
        # =============================================================
        print("\n--- 7. SEASONALITY STRENGTH SUMMARY ---")
        
        # Calculate overall seasonality scores
        yearly_score = 0
        weekly_score = 0
        
        # From statistical tests
        if yearly_significant:
            yearly_score += yearly_eta_squared * 30  # Weight by effect size
        if weekly_significant:
            weekly_score += weekly_eta_squared * 30
        
        # From decomposition
        if 'yearly' in product_analysis.get('decomposition', {}):
            yearly_score += product_analysis['decomposition']['yearly'] * 20
        if 'weekly' in product_analysis.get('decomposition', {}):
            weekly_score += product_analysis['decomposition']['weekly'] * 20
        
        # From spectral analysis
        for spec in product_analysis.get('spectral', []):
            if 350 <= spec['period'] <= 380:  # Yearly
                yearly_score += min(spec['power'] * 1e6, 10)  # Cap at 10
            elif 6 <= spec['period'] <= 8:  # Weekly
                weekly_score += min(spec['power'] * 1e6, 10)
        
        # Normalize scores
        yearly_score = min(yearly_score, 100)
        weekly_score = min(weekly_score, 100)
        
        print(f"Yearly seasonality strength: {yearly_score:.1f}/100")
        print(f"Weekly seasonality strength: {weekly_score:.1f}/100")
        
        # Classification
        yearly_class = "STRONG" if yearly_score >= 70 else "MODERATE" if yearly_score >= 30 else "WEAK"
        weekly_class = "STRONG" if weekly_score >= 70 else "MODERATE" if weekly_score >= 30 else "WEAK"
        
        print(f"Yearly seasonality: {yearly_class}")
        print(f"Weekly seasonality: {weekly_class}")
        
        product_analysis['seasonality_strength'] = {
            'yearly_score': yearly_score,
            'weekly_score': weekly_score,
            'yearly_class': yearly_class,
            'weekly_class': weekly_class
        }
        
        seasonality_analysis[product] = product_analysis
    
    return seasonality_analysis

def model_recommendations_based_on_seasonality(seasonality_analysis):
    """
    Recommend the best modeling approach based on detected seasonality patterns
    """
    print("\n" + "="*80)
    print("MODEL RECOMMENDATIONS BASED ON SEASONALITY ANALYSIS")
    print("="*80)
    
    # Analyze patterns across all products
    yearly_strong = []
    yearly_moderate = []
    yearly_weak = []
    weekly_strong = []
    weekly_moderate = []
    weekly_weak = []
    
    categories = {}
    
    for product, analysis in seasonality_analysis.items():
        if 'seasonality_strength' not in analysis:
            continue
            
        category = analysis['category']
        if category not in categories:
            categories[category] = {
                'products': [],
                'yearly_scores': [],
                'weekly_scores': [],
                'yearly_patterns': [],
                'weekly_patterns': []
            }
        
        categories[category]['products'].append(product)
        
        strength = analysis['seasonality_strength']
        categories[category]['yearly_scores'].append(strength['yearly_score'])
        categories[category]['weekly_scores'].append(strength['weekly_score'])
        
        # Classify by strength
        if strength['yearly_class'] == 'STRONG':
            yearly_strong.append(product)
        elif strength['yearly_class'] == 'MODERATE':
            yearly_moderate.append(product)
        else:
            yearly_weak.append(product)
            
        if strength['weekly_class'] == 'STRONG':
            weekly_strong.append(product)
        elif strength['weekly_class'] == 'MODERATE':
            weekly_moderate.append(product)
        else:
            weekly_weak.append(product)
    
    total_products = len(seasonality_analysis)
    
    print(f"\n--- SEASONALITY DISTRIBUTION ---")
    print(f"Total products analyzed: {total_products}")
    print(f"\nYearly seasonality:")
    print(f"  Strong: {len(yearly_strong)} products ({len(yearly_strong)/total_products*100:.1f}%)")
    print(f"  Moderate: {len(yearly_moderate)} products ({len(yearly_moderate)/total_products*100:.1f}%)")
    print(f"  Weak: {len(yearly_weak)} products ({len(yearly_weak)/total_products*100:.1f}%)")
    
    print(f"\nWeekly seasonality:")
    print(f"  Strong: {len(weekly_strong)} products ({len(weekly_strong)/total_products*100:.1f}%)")
    print(f"  Moderate: {len(weekly_moderate)} products ({len(weekly_moderate)/total_products*100:.1f}%)")
    print(f"  Weak: {len(weekly_weak)} products ({len(weekly_weak)/total_products*100:.1f}%)")
    
    print(f"\n--- CATEGORY-LEVEL SEASONALITY ---")
    category_consistency = {}
    
    for category, data in categories.items():
        avg_yearly = np.mean(data['yearly_scores'])
        avg_weekly = np.mean(data['weekly_scores'])
        std_yearly = np.std(data['yearly_scores'])
        std_weekly = np.std(data['weekly_scores'])
        
        print(f"\n{category.upper()}:")
        print(f"  Products: {len(data['products'])}")
        print(f"  Avg yearly score: {avg_yearly:.1f} ± {std_yearly:.1f}")
        print(f"  Avg weekly score: {avg_weekly:.1f} ± {std_weekly:.1f}")
        
        # Check consistency within category
        yearly_cv = std_yearly / avg_yearly if avg_yearly > 0 else 0
        weekly_cv = std_weekly / avg_weekly if avg_weekly > 0 else 0
        
        yearly_consistent = yearly_cv < 0.5  # Less than 50% variation
        weekly_consistent = weekly_cv < 0.5
        
        print(f"  Yearly consistency: {'YES' if yearly_consistent else 'NO'} (CV={yearly_cv:.2f})")
        print(f"  Weekly consistency: {'YES' if weekly_consistent else 'NO'} (CV={weekly_cv:.2f})")
        
        category_consistency[category] = {
            'yearly_consistent': yearly_consistent,
            'weekly_consistent': weekly_consistent,
            'avg_yearly': avg_yearly,
            'avg_weekly': avg_weekly
        }
    
    print(f"\n" + "="*80)
    print("🎯 MODEL RECOMMENDATIONS")
    print("="*80)
    
    # Check if we can use simple approaches
    strong_seasonality_pct = (len(yearly_strong) + len(weekly_strong)) / (total_products * 2) * 100
    
    if strong_seasonality_pct < 30:
        print(f"\n1️⃣ RECOMMENDATION: SIMPLE MODELS")
        print(f"   → Low seasonality detected ({strong_seasonality_pct:.1f}% strong seasonal patterns)")
        print(f"   → Consider: Linear Regression, ARIMA, Exponential Smoothing")
        print(f"   → Avoid: Complex seasonal models like Prophet, SARIMA")
        
    elif all(cat['yearly_consistent'] and cat['weekly_consistent'] for cat in category_consistency.values()):
        print(f"\n1️⃣ RECOMMENDATION: CATEGORY-BASED MODELS")
        print(f"   → Consistent seasonality within categories")
        print(f"   → Use {len(categories)} models (one per category)")
        print(f"   → Suggested models: SARIMA, Prophet, Seasonal Exponential Smoothing")
        
        print(f"\n   Category-specific model settings:")
        for category, data in category_consistency.items():
            yearly_setting = data['avg_yearly'] >= 50
            weekly_setting = data['avg_weekly'] >= 50
            
            if data['avg_yearly'] >= 70 or data['avg_weekly'] >= 70:
                model_type = "Prophet or SARIMA (complex seasonality)"
            elif data['avg_yearly'] >= 30 or data['avg_weekly'] >= 30:
                model_type = "Seasonal Exponential Smoothing"
            else:
                model_type = "Simple Exponential Smoothing or ARIMA"
            
            print(f"     • {category}: {model_type}")
            print(f"       - Yearly seasonality: {'Include' if yearly_setting else 'Skip'}")
            print(f"       - Weekly seasonality: {'Include' if weekly_setting else 'Skip'}")
    
    else:
        print(f"\n1️⃣ RECOMMENDATION: HYBRID APPROACH")
        print(f"   → Variable seasonality patterns across products")
        print(f"   → Use different models for different seasonal strength groups")
        
        print(f"\n   Model assignment by seasonality strength:")
        print(f"   • Strong seasonal products ({len(yearly_strong + weekly_strong)} products):")
        print(f"     → Prophet, SARIMA, or Seasonal Exponential Smoothing")
        print(f"   • Moderate seasonal products ({len(yearly_moderate + weekly_moderate)} products):")
        print(f"     → Seasonal Exponential Smoothing or simple SARIMA")
        print(f"   • Weak seasonal products ({len(yearly_weak + weekly_weak)} products):")
        print(f"     → ARIMA, Linear Regression, or Simple Exponential Smoothing")
    
    print(f"\n2️⃣ SPECIFIC MODEL RECOMMENDATIONS:")
    
    print(f"\n   🏆 PROPHET:")
    print(f"   ✅ Use if: Strong yearly/weekly seasonality, want automatic detection")
    print(f"   ✅ Good for: {len(yearly_strong + weekly_strong)} products with strong seasonality")
    print(f"   ❌ Avoid for: Products with weak seasonality (computational overhead)")
    
    print(f"\n   📈 SARIMA:")
    print(f"   ✅ Use if: Clear seasonal periods, stationary data after differencing")
    print(f"   ✅ Good for: Products with moderate to strong seasonality")
    print(f"   ❌ Avoid for: Very irregular patterns, non-stationary trends")
    
    print(f"\n   📊 SEASONAL EXPONENTIAL SMOOTHING:")
    print(f"   ✅ Use if: Simple seasonality, fast computation needed")
    print(f"   ✅ Good for: Most products with any seasonal pattern")
    print(f"   ❌ Avoid for: Complex multiple seasonalities")
    
    print(f"\n   📉 ARIMA/LINEAR REGRESSION:")
    print(f"   ✅ Use if: Weak seasonality, trend-focused")
    print(f"   ✅ Good for: {len(yearly_weak + weekly_weak)} products with weak seasonality")
    print(f"   ❌ Avoid for: Strong seasonal patterns")
    
    print(f"\n3️⃣ IMPLEMENTATION STRATEGY:")
    
    # Check computational complexity
    if total_products >= 1000:
        print(f"\n   ⚡ FOR {total_products} PRODUCTS (LARGE SCALE):")
        print(f"   1. Start with category-level models ({len(categories)} models)")
        print(f"   2. Use Seasonal Exponential Smoothing for speed")
        print(f"   3. Upgrade to Prophet only for top-performing products")
        print(f"   4. Consider parallel processing")
    else:
        print(f"\n   🎯 FOR {total_products} PRODUCTS (MANAGEABLE SCALE):")
        print(f"   1. Use Prophet for products with strong seasonality")
        print(f"   2. Use SARIMA for moderate seasonality")
        print(f"   3. Use simple models for weak seasonality")
        print(f"   4. Individual models are feasible")

def create_seasonality_plots(seasonality_analysis, save_plots=True):
    """
    Create visualizations of seasonality patterns
    """
    print("\n--- CREATING SEASONALITY VISUALIZATION ---")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    products = []
    categories = []
    yearly_scores = []
    weekly_scores = []
    
    for product, analysis in seasonality_analysis.items():
        if 'seasonality_strength' in analysis:
            products.append(product)
            categories.append(analysis['category'])
            yearly_scores.append(analysis['seasonality_strength']['yearly_score'])
            weekly_scores.append(analysis['seasonality_strength']['weekly_score'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Seasonality Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Seasonality scores by product
    x_pos = range(len(products))
    axes[0,0].bar([x-0.2 for x in x_pos], yearly_scores, width=0.4, label='Yearly', alpha=0.7)
    axes[0,0].bar([x+0.2 for x in x_pos], weekly_scores, width=0.4, label='Weekly', alpha=0.7)
    axes[0,0].set_title('Seasonality Scores by Product')
    axes[0,0].set_xlabel('Product')
    axes[0,0].set_ylabel('Seasonality Score')
    axes[0,0].legend()
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels([p.split('_')[-1] for p in products], rotation=45)
    
    # 2. Seasonality distribution
    bins = [0, 30, 70, 100]
    labels = ['Weak', 'Moderate', 'Strong']
    yearly_hist, _ = np.histogram(yearly_scores, bins=bins)
    weekly_hist, _ = np.histogram(weekly_scores, bins=bins)
    
    x_labels = range(len(labels))
    axes[0,1].bar([x-0.2 for x in x_labels], yearly_hist, width=0.4, label='Yearly', alpha=0.7)
    axes[0,1].bar([x+0.2 for x in x_labels], weekly_hist, width=0.4, label='Weekly', alpha=0.7)
    axes[0,1].set_title('Seasonality Strength Distribution')
    axes[0,1].set_xlabel('Seasonality Strength')
    axes[0,1].set_ylabel('Number of Products')
    axes[0,1].set_xticks(x_labels)
    axes[0,1].set_xticklabels(labels)
    axes[0,1].legend()
    
    # 3. Seasonality by category
    category_data = {}
    for cat, yearly, weekly in zip(categories, yearly_scores, weekly_scores):
        if cat not in category_data:
            category_data[cat] = {'yearly': [], 'weekly': []}
        category_data[cat]['yearly'].append(yearly)
        category_data[cat]['weekly'].append(weekly)
    
    cat_names = list(category_data.keys())
    cat_yearly_means = [np.mean(category_data[cat]['yearly']) for cat in cat_names]
    cat_weekly_means = [np.mean(category_data[cat]['weekly']) for cat in cat_names]
    
    x_cat = range(len(cat_names))
    axes[0,2].bar([x-0.2 for x in x_cat], cat_yearly_means, width=0.4, label='Yearly', alpha=0.7)
    axes[0,2].bar([x+0.2 for x in x_cat], cat_weekly_means, width=0.4, label='Weekly', alpha=0.7)
    axes[0,2].set_title('Average Seasonality by Category')
    axes[0,2].set_xlabel('Category')
    axes[0,2].set_ylabel('Average Seasonality Score')
    axes[0,2].legend()
    axes[0,2].set_xticks(x_cat)
    axes[0,2].set_xticklabels(cat_names, rotation=45)
    
    # 4. Scatter plot: Yearly vs Weekly seasonality
    scatter = axes[1,0].scatter(yearly_scores, weekly_scores, c=[hash(cat) for cat in categories], alpha=0.7)
    axes[1,0].set_title('Yearly vs Weekly Seasonality')
    axes[1,0].set_xlabel('Yearly Seasonality Score')
    axes[1,0].set_ylabel('Weekly Seasonality Score')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add quadrant lines
    axes[1,0].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=50, color='r', linestyle='--', alpha=0.5)
    
    # 5. Seasonality heatmap by category
    if len(cat_names) > 1:
        heatmap_data = np.array([cat_yearly_means, cat_weekly_means])
        im = axes[1,1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_title('Seasonality Heatmap by Category')
        axes[1,1].set_xticks(range(len(cat_names)))
        axes[1,1].set_xticklabels(cat_names, rotation=45)
        axes[1,1].set_yticks([0, 1])
        axes[1,1].set_yticklabels(['Yearly', 'Weekly'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1,1])
        cbar.set_label('Seasonality Score')
        
        # Add text annotations
        for i in range(len(cat_names)):
            for j in range(2):
                text = axes[1,1].text(i, j, f'{heatmap_data[j, i]:.1f}',
                                    ha="center", va="center", color="black", fontweight='bold')
    
    # 6. Model recommendation summary
    # Count products by seasonality strength
    strong_count = sum(1 for y, w in zip(yearly_scores, weekly_scores) if y >= 70 or w >= 70)
    moderate_count = sum(1 for y, w in zip(yearly_scores, weekly_scores) if (30 <= y < 70 or 30 <= w < 70) and not (y >= 70 or w >= 70))
    weak_count = len(yearly_scores) - strong_count - moderate_count
    
    model_counts = [strong_count, moderate_count, weak_count]
    model_labels = ['Prophet/SARIMA\n(Strong)', 'Seasonal ETS\n(Moderate)', 'ARIMA/Linear\n(Weak)']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    axes[1,2].pie(model_counts, labels=model_labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1,2].set_title('Recommended Models Distribution')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('seasonality_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Seasonality plots saved as 'seasonality_analysis.png'")
    
    plt.show()

if __name__ == "__main__":
    # Choose analysis type
    print("Choose analysis type:")
    print("1. Cross-validation analysis (recommended for parameter tuning)")
    print("2. Standard Prophet modeling")
    print("3. Individual product analysis")
    print("4. Seasonality pattern analysis (detect different seasonalities)")
    print("5. Comprehensive seasonality detection (NO Prophet - pure statistical analysis)")
    
    choice = input("Enter choice (1, 2, 3, 4, or 5): ").strip()
    
    if choice == "1":
        # Run cross-validation analysis
        print("Running cross-validation analysis...")
        product_results, category_results = main(run_cross_validation=True)
    elif choice == "2":
        # Run standard analysis
        print("Running standard Prophet analysis...")
        model, forecast, metrics = main(run_cross_validation=False)
    elif choice == "3":
        # Run individual product analysis
        print("Running individual product analysis...")
        # Load data first
        df = load_and_prepare_data('daily_demand_by_product_modern.csv')
        individual_results = analyze_individual_products(df, n_products=5)
        
        # Compare and plot results
        compare_product_characteristics(individual_results)
        plot_product_comparison(individual_results)
    elif choice == "4":
        # Run seasonality analysis
        print("Running seasonality pattern analysis...")
        # Load data first
        df = load_and_prepare_data('daily_demand_by_product_modern.csv')
        
        # Analyze seasonality patterns
        seasonality_results = analyze_seasonality_patterns(df, n_products=10)
        
        # Compare seasonality across products
        category_seasonality = compare_seasonality_across_products(seasonality_results)
        
        # Get recommendations
        prophet_seasonality_recommendations(seasonality_results, category_seasonality)
        
    elif choice == "5":
        # Run comprehensive seasonality detection (no Prophet)
        print("Running comprehensive seasonality detection (without Prophet)...")
        print("This will install required packages if needed...")
        
        # Install statsmodels if not available
        try:
            import statsmodels
        except ImportError:
            print("Installing statsmodels package...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
            import statsmodels
        
        # Load data first
        df = load_and_prepare_data('daily_demand_by_product_modern.csv')
        
        # Comprehensive seasonality analysis
        seasonality_analysis = comprehensive_seasonality_detection(df, n_products=10)
        
        # Get model recommendations
        category_consistency = model_recommendations_based_on_seasonality(seasonality_analysis)
        
        # Create visualizations
        create_seasonality_plots(seasonality_analysis, save_plots=True)
        
    else:
        print("Invalid choice. Please run the program again and select 1, 2, 3, 4, or 5.")