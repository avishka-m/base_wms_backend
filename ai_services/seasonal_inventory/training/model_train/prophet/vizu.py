import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from prophet import Prophet

# Load your dataset
df = pd.read_csv('daily_demand_by_product_modern.csv')

# Convert date column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Option 1: Plot total demand across all products by date
daily_total = df.groupby('ds')['y'].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_total['ds'], daily_total['y'])
ax.set_xlabel('Date')
ax.set_ylabel('Total Daily Demand')
ax.set_title('Total Daily Demand Across All Products')

# Format x-axis
plt.xticks(rotation=45)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# Option 2: Plot demand for a specific product
print("Available products (first 10):")
print(df['product_id'].unique()[:10])

# Let's plot the first product as an example
first_product = df['product_id'].iloc[0]
product_data = df[df['product_id'] == first_product].copy()

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(product_data['ds'], product_data['y'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Demand')
ax2.set_title(f'Daily Demand for {first_product}')

fig2.autofmt_xdate()  # This already handles rotation and formatting
plt.tight_layout()
plt.show()

# Option 3: Plot by category
print("\nAvailable categories:")
print(df['category'].unique())

category_totals = df.groupby(['ds', 'category'])['y'].sum().reset_index()
print(category_totals)
fig3, ax3 = plt.subplots(figsize=(14, 8))
categories = df['category'].unique()

for category in categories:
    cat_data = category_totals[category_totals['category'] == category]
    ax3.plot(cat_data['ds'], cat_data['y'], label=category, alpha=0.7)

ax3.set_xlabel('Date')
ax3.set_ylabel('Total Daily Demand')
ax3.set_title('Daily Demand by Product Category')
ax3.legend()

plt.xticks(rotation=45)
fig3.autofmt_xdate()
plt.tight_layout()
plt.show()

def show_prophet_defaults():
    """Display Prophet's default parameters"""
    print("="*50)
    print("PROPHET DEFAULT PARAMETERS")
    print("="*50)
    
    # Create a default Prophet model to inspect parameters
    default_model = Prophet()
    
    print("SEASONALITY PARAMETERS:")
    print(f"  yearly_seasonality: {default_model.yearly_seasonality}")
    print(f"  weekly_seasonality: {default_model.weekly_seasonality}")
    print(f"  daily_seasonality: {default_model.daily_seasonality}")
    print(f"  seasonality_mode: '{default_model.seasonality_mode}'")
    
    print("\nPRIOR SCALE PARAMETERS:")
    print(f"  changepoint_prior_scale: {default_model.changepoint_prior_scale}")
    print(f"  seasonality_prior_scale: {default_model.seasonality_prior_scale}")
    print(f"  holidays_prior_scale: {default_model.holidays_prior_scale}")
    
    print("\nCHANGEPOINT PARAMETERS:")
    print(f"  n_changepoints: {default_model.n_changepoints}")
    print(f"  changepoint_range: {default_model.changepoint_range}")
    
    print("\nOTHER PARAMETERS:")
    print(f"  growth: '{default_model.growth}'")
    print(f"  interval_width: {default_model.interval_width}")
    print(f"  uncertainty_samples: {default_model.uncertainty_samples}")
    print(f"  mcmc_samples: {default_model.mcmc_samples}")
    
    print("\n" + "="*50)
    print("PARAMETER EXPLANATIONS:")
    print("="*50)
    print("changepoint_prior_scale: Controls trend flexibility")
    print("  - Lower (0.01): Very stable trend, few changes")
    print("  - Default (0.05): Moderate flexibility") 
    print("  - Higher (0.5): Very flexible, many trend changes")
    
    print("\nseasonality_prior_scale: Controls seasonality strength")
    print("  - Lower (0.1): Weak seasonal effects")
    print("  - Default (10.0): Strong seasonal effects")
    print("  - Higher (100.0): Very strong seasonal effects")
    
    print("\nseasonality_mode: How seasonality affects the trend")
    print("  - 'additive' (default): Seasonal effects stay constant")
    print("  - 'multiplicative': Seasonal effects scale with trend")

# Add this to your visualization script
show_prophet_defaults()