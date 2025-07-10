# """
# Web Scraper for External Market Data Collection

# This module scrapes real-time market data, economic indicators, and seasonal trends
# from various web sources to enhance inventory forecasting accuracy.
# """

# import requests
# import pandas as pd
# import logging
# from typing import Dict, List
# from datetime import datetime, timedelta
# import time
# from pathlib import Path

# from config import WEATHER_API_KEY, FRED_API_KEY, ALPHA_VANTAGE_API_KEY, CACHE_DIR

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class WebDataScraper:
#     """
#     Scrapes external data sources for seasonal inventory forecasting.
#     """
    
#     def __init__(self, cache_dir: str = CACHE_DIR):
#         """
#         Initialize the web scraper.
        
#         Args:
#             cache_dir: Directory to cache scraped data
#         """
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
        
#         # Session for HTTP requests
#         self.session = requests.Session()
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
        
#         # API keys
#         self.weather_api_key = WEATHER_API_KEY
#         self.fred_api_key = FRED_API_KEY
#         self.alpha_vantage_key = ALPHA_VANTAGE_API_KEY
        
#         logger.info("🕸️ Web scraper initialized")
    
#     def get_economic_indicators(self, indicators: List[str] = None, 
#                                start_date: str = None, 
#                                end_date: str = None) -> pd.DataFrame:
#         """
#         Fetch economic indicators from FRED (Federal Reserve Economic Data).
        
#         Args:
#             indicators: List of FRED series IDs
#             start_date: Start date in YYYY-MM-DD format
#             end_date: End date in YYYY-MM-DD format
            
#         Returns:
#             DataFrame with economic indicators
#         """
#         if not self.fred_api_key:
#             logger.warning("FRED API key not provided, skipping economic data")
#             return pd.DataFrame()
        
#         if indicators is None:
#             indicators = [
#                 'GDP',           # Gross Domestic Product
#                 'UNRATE',        # Unemployment Rate
#                 'CPIAUCSL',      # Consumer Price Index
#                 'PAYEMS',        # Nonfarm Payrolls
#                 'HOUST',         # Housing Starts
#                 'RSXFS',         # Retail Sales
#                 'INDPRO',        # Industrial Production Index
#                 'UMCSENT'        # Consumer Sentiment
#             ]
        
#         if not start_date:
#             start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
#         if not end_date:
#             end_date = datetime.now().strftime('%Y-%m-%d')
        
#         economic_data = []
        
#         for indicator in indicators:
#             try:
#                 logger.info(f"📈 Fetching {indicator} from FRED")
                
#                 url = "https://api.stlouisfed.org/fred/series/observations"
#                 params = {
#                     'series_id': indicator,
#                     'api_key': self.fred_api_key,
#                     'file_type': 'json',
#                     'observation_start': start_date,
#                     'observation_end': end_date
#                 }
                
#                 response = self.session.get(url, params=params)
#                 response.raise_for_status()
                
#                 data = response.json()
#                 observations = data.get('observations', [])
                
#                 for obs in observations:
#                     if obs['value'] != '.':  # FRED uses '.' for missing values
#                         economic_data.append({
#                             'ds': pd.to_datetime(obs['date']),
#                             'indicator': indicator,
#                             'value': float(obs['value'])
#                         })
                
#                 # Rate limiting
#                 time.sleep(0.1)
                
#             except Exception as e:
#                 logger.error(f"❌ Failed to fetch {indicator}: {e}")
#                 continue
        
#         if economic_data:
#             df = pd.DataFrame(economic_data)
#             df = df.pivot(index='ds', columns='indicator', values='value')
#             df = df.fillna(method='ffill')  # Forward fill missing values
            
#             logger.info(f"📊 Retrieved {len(df)} economic data points")
            
#             # Cache the data
#             cache_file = self.cache_dir / "economic_indicators.csv"
#             df.to_csv(cache_file)
#             logger.info(f"💾 Cached economic data to {cache_file}")
            
#             return df
        
#         return pd.DataFrame()
    
#     def get_weather_data(self, cities: List[str] = None, 
#                         days_back: int = 365) -> pd.DataFrame:
#         """
#         Fetch historical weather data that might affect inventory patterns.
        
#         Args:
#             cities: List of city names to get weather for
#             days_back: Number of days of historical data
            
#         Returns:
#             DataFrame with weather data
#         """
#         if not self.weather_api_key:
#             logger.warning("Weather API key not provided, skipping weather data")
#             return pd.DataFrame()
        
#         if cities is None:
#             cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
        
#         weather_data = []
        
#         for city in cities:
#             try:
#                 logger.info(f"🌤️ Fetching weather data for {city}")
                
#                 # Get current weather and basic historical
#                 url = "http://api.openweathermap.org/data/2.5/weather"
#                 params = {
#                     'q': city,
#                     'appid': self.weather_api_key,
#                     'units': 'metric'
#                 }
                
#                 response = self.session.get(url, params=params)
#                 response.raise_for_status()
                
#                 data = response.json()
                
#                 weather_data.append({
#                     'ds': pd.to_datetime(datetime.now().date()),
#                     'city': city,
#                     'temperature': data['main']['temp'],
#                     'humidity': data['main']['humidity'],
#                     'pressure': data['main']['pressure'],
#                     'weather_main': data['weather'][0]['main'],
#                     'weather_desc': data['weather'][0]['description']
#                 })
                
#                 time.sleep(0.1)  # Rate limiting
                
#             except Exception as e:
#                 logger.error(f"❌ Failed to fetch weather for {city}: {e}")
#                 continue
        
#         if weather_data:
#             df = pd.DataFrame(weather_data)
#             logger.info(f"🌡️ Retrieved weather data for {len(cities)} cities")
            
#             # Cache the data
#             cache_file = self.cache_dir / "weather_data.csv"
#             df.to_csv(cache_file, index=False)
            
#             return df
        
#         return pd.DataFrame()
    
#     def get_retail_trends(self, keywords: List[str] = None) -> pd.DataFrame:
#         """
#         Scrape retail trend data from various sources.
        
#         Args:
#             keywords: List of product keywords to track
            
#         Returns:
#             DataFrame with trend data
#         """
#         if keywords is None:
#             keywords = [
#                 'electronics', 'clothing', 'home goods', 'books', 
#                 'toys', 'sports equipment', 'garden supplies'
#             ]
        
#         trend_data = []
        
#         try:
#             # Scrape Google Trends (simplified approach)
#             logger.info("📊 Scraping retail trend data")
            
#             for keyword in keywords:
#                 # This is a simplified example - in practice you'd use
#                 # the official Google Trends API or pytrends library
#                 trend_data.append({
#                     'ds': pd.to_datetime(datetime.now().date()),
#                     'keyword': keyword,
#                     'trend_score': 50 + (len(keyword) % 50),  # Mock data
#                     'category': 'retail'
#                 })
            
#             if trend_data:
#                 df = pd.DataFrame(trend_data)
#                 logger.info(f"📈 Retrieved trend data for {len(keywords)} keywords")
                
#                 # Cache the data
#                 cache_file = self.cache_dir / "retail_trends.csv"
#                 df.to_csv(cache_file, index=False)
                
#                 return df
                
#         except Exception as e:
#             logger.error(f"❌ Failed to scrape retail trends: {e}")
        
#         return pd.DataFrame()
    
#     def get_holiday_calendar(self, years: List[int] = None, 
#                            countries: List[str] = None) -> pd.DataFrame:
#         """
#         Scrape holiday calendar data from timeanddate.com or similar sources.
        
#         Args:
#             years: List of years to get holidays for
#             countries: List of country codes
            
#         Returns:
#             DataFrame with holiday information
#         """
#         if years is None:
#             current_year = datetime.now().year
#             years = [current_year - 1, current_year, current_year + 1]
        
#         if countries is None:
#             countries = ['US', 'GB', 'BR', 'IN']
        
#         holidays_data = []
        
#         # Static holiday data (in practice, you'd scrape from timeanddate.com)
#         static_holidays = {
#             'US': [
#                 {'name': 'New Year', 'date': '01-01'},
#                 {'name': 'Independence Day', 'date': '07-04'},
#                 {'name': 'Thanksgiving', 'date': '11-23'},  # Approximate
#                 {'name': 'Christmas', 'date': '12-25'},
#                 {'name': 'Black Friday', 'date': '11-24'},  # Approximate
#                 {'name': 'Cyber Monday', 'date': '11-27'}   # Approximate
#             ],
#             'GB': [
#                 {'name': 'New Year', 'date': '01-01'},
#                 {'name': 'Christmas', 'date': '12-25'},
#                 {'name': 'Boxing Day', 'date': '12-26'}
#             ]
#         }
        
#         try:
#             logger.info("📅 Building holiday calendar")
            
#             for year in years:
#                 for country in countries:
#                     if country in static_holidays:
#                         for holiday in static_holidays[country]:
#                             holidays_data.append({
#                                 'ds': pd.to_datetime(f"{year}-{holiday['date']}"),
#                                 'holiday': holiday['name'],
#                                 'country': country,
#                                 'year': year
#                             })
            
#             if holidays_data:
#                 df = pd.DataFrame(holidays_data)
#                 df = df.sort_values('ds').reset_index(drop=True)
                
#                 logger.info(f"📅 Generated {len(df)} holiday entries")
                
#                 # Cache the data
#                 cache_file = self.cache_dir / "holiday_calendar.csv"
#                 df.to_csv(cache_file, index=False)
                
#                 return df
                
#         except Exception as e:
#             logger.error(f"❌ Failed to build holiday calendar: {e}")
        
#         return pd.DataFrame()
    
#     def scrape_all_external_data(self) -> Dict[str, pd.DataFrame]:
#         """
#         Run all data scraping operations and return combined results.
        
#         Returns:
#             Dictionary containing all scraped datasets
#         """
#         logger.info("🚀 Starting comprehensive external data scraping")
        
#         results = {}
        
#         # Economic indicators
#         logger.info("💰 Scraping economic indicators...")
#         results['economic'] = self.get_economic_indicators()
        
#         # Weather data
#         logger.info("🌤️ Scraping weather data...")
#         results['weather'] = self.get_weather_data()
        
#         # Retail trends
#         logger.info("📊 Scraping retail trends...")
#         results['trends'] = self.get_retail_trends()
        
#         # Holiday calendar
#         logger.info("📅 Building holiday calendar...")
#         results['holidays'] = self.get_holiday_calendar()
        
#         # Summary
#         total_records = sum(len(df) for df in results.values() if not df.empty)
#         logger.info(f"✅ External data scraping complete: {total_records} total records")
        
#         return results
    
#     def create_external_features_dataset(self) -> pd.DataFrame:
#         """
#         Create a comprehensive external features dataset for Prophet model.
        
#         Returns:
#             Combined DataFrame with all external features
#         """
#         logger.info("🔄 Creating comprehensive external features dataset")
        
#         # Get all external data
#         data = self.scrape_all_external_data()
        
#         # Start with a date range
#         start_date = datetime.now() - timedelta(days=365*2)
#         end_date = datetime.now() + timedelta(days=365)
#         date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
#         features_df = pd.DataFrame({'ds': date_range})
        
#         # Merge economic indicators (monthly data, forward fill)
#         if not data['economic'].empty:
#             economic_daily = data['economic'].resample('D').ffill()
#             features_df = features_df.merge(
#                 economic_daily.reset_index(), 
#                 on='ds', 
#                 how='left'
#             )
        
#         # Add weather features (daily data)
#         if not data['weather'].empty:
#             weather_agg = data['weather'].groupby('ds').agg({
#                 'temperature': 'mean',
#                 'humidity': 'mean',
#                 'pressure': 'mean'
#             }).reset_index()
            
#             features_df = features_df.merge(weather_agg, on='ds', how='left')
        
#         # Add holiday flags
#         if not data['holidays'].empty:
#             holiday_flags = data['holidays'].groupby('ds')['holiday'].first().reset_index()
#             holiday_flags['is_holiday'] = True
#             features_df = features_df.merge(holiday_flags, on='ds', how='left')
#             features_df['is_holiday'] = features_df['is_holiday'].fillna(False)
        
#         # Add seasonal features
#         features_df['month'] = features_df['ds'].dt.month
#         features_df['quarter'] = features_df['ds'].dt.quarter
#         features_df['day_of_week'] = features_df['ds'].dt.dayofweek
#         features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6])
#         features_df['is_month_end'] = features_df['ds'].dt.is_month_end
#         features_df['is_quarter_end'] = features_df['ds'].dt.is_quarter_end
        
#         # Fill missing values
#         features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
#         logger.info(f"📊 Created external features dataset with {len(features_df)} rows and {len(features_df.columns)} features")
        
#         # Save the comprehensive dataset
#         output_file = self.cache_dir / "external_features.csv"
#         features_df.to_csv(output_file, index=False)
#         logger.info(f"💾 Saved external features to {output_file}")
        
#         return features_df


# def main():
#     """Main function to demonstrate web scraping functionality."""
#     print("🕸️ Web Data Scraper for Seasonal Inventory Prediction")
#     print("=" * 60)
    
#     # Initialize scraper
#     scraper = WebDataScraper()
    
#     # Scrape all external data
#     print("\n🔄 Scraping external data...")
#     results = scraper.scrape_all_external_data()
    
#     # Show results summary
#     print("\n📊 Scraping Results:")
#     for source, df in results.items():
#         if not df.empty:
#             print(f"   {source.title()}: {len(df)} records")
#             print(f"   Columns: {list(df.columns)}")
#         else:
#             print(f"   {source.title()}: No data (check API keys)")
#         print()
    
#     # Create comprehensive features dataset
#     print("🔧 Creating external features dataset...")
#     features_df = scraper.create_external_features_dataset()
    
#     print("\nExternal features dataset created:")
#     print(f"   Shape: {features_df.shape}")
#     print(f"   Date range: {features_df['ds'].min()} to {features_df['ds'].max()}")
#     print(f"   Features: {list(features_df.columns)}")
    
#     # Preview the data
#     print("\n👀 Sample external features:")
#     print(features_df.head().to_string())


# if __name__ == "__main__":
#     main()
