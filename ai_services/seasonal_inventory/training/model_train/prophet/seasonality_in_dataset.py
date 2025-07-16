import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_csv('daily_demand_by_product_modern.csv')  # must have 'ds' and 'y'
# df['ds'] = pd.to_datetime(df['ds'])

# plt.figure(figsize=(10, 4))
# plt.plot(df['ds'], df['y'])
# plt.title('Raw Time Series')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()

# Example: filter one product
product_id = 'PROD_2026_BOOK_1875'
df_one = df[df['product_id'] == product_id].copy()

# Ensure it's sorted by date
df_one = df_one.sort_values('ds')
# Set 'ds' as index
df_one.set_index('ds', inplace=True)

# Convert ds to datetime
df_one.index = pd.to_datetime(df_one.index)

# Perform seasonal decomposition on the single product
result = seasonal_decompose(df_one['y'], model='additive', period=365)  # use 7 for weekly, 12 for monthly, etc.

result.plot()
plt.tight_layout()
plt.show()

