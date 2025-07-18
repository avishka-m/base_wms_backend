"""
Modern E-commerce Data Generator for Seasonal Inventory Prediction

Generates realistic 2022-2024 e-commerce transaction data that reflects:
- Current consumer behavior patterns
- Modern seasonality (Black Friday, Prime Day, etc.)
- COVID-19 impact and recovery
- Supply chain disruptions
- Inflation effects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModernEcommerceDataGenerator:
    """Generate realistic modern e-commerce data for training Prophet models"""
    
    def __init__(self):
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 12, 31)
        self.categories = {
            "electronics": {"base_demand": 15, "volatility": 0.3, "growth_rate": 0.1},
            "clothing": {"base_demand": 25, "volatility": 0.4, "growth_rate": 0.05},
            "home_garden": {"base_demand": 12, "volatility": 0.25, "growth_rate": 0.08},
            "books_media": {"base_demand": 8, "volatility": 0.2, "growth_rate": -0.02},
            "health_beauty": {"base_demand": 18, "volatility": 0.35, "growth_rate": 0.12},
            "sports_outdoors": {"base_demand": 10, "volatility": 0.45, "growth_rate": 0.06}
        }
        
    def generate_modern_dataset(self, num_products: int = 1000, output_path: str = None) -> pd.DataFrame:
        """Generate comprehensive modern e-commerce dataset"""
        
        logger.info(f"🔄 Generating modern e-commerce data for {num_products} products...")
        
        # Generate date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        all_data = []
        
        # Generate products across categories
        for product_idx in range(num_products):
            category = random.choice(list(self.categories.keys()))
            product_id = f"PROD_{2022 + product_idx // 400}_{category[:4].upper()}_{product_idx:04d}"
            
            product_data = self._generate_product_timeline(
                product_id, 
                category, 
                date_range
            )
            all_data.extend(product_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values(['product_id', 'ds'])
        
        logger.info(f"✅ Generated {len(df)} records for {num_products} products")
        
        # Save if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved dataset to {output_file}")
        
        return df
    
    def _generate_product_timeline(self, product_id: str, category: str, date_range: pd.DatetimeIndex) -> list:
        """Generate realistic demand timeline for a single product"""
        
        category_config = self.categories[category]
        base_demand = category_config["base_demand"]
        volatility = category_config["volatility"]
        growth_rate = category_config["growth_rate"]
        
        timeline_data = []
        
        for date in date_range:
            # Base demand with growth trend
            days_since_start = (date - self.start_date).days
            trend_multiplier = 1 + (growth_rate * days_since_start / 365.0)
            
            # Seasonal patterns
            seasonal_demand = self._calculate_seasonal_multiplier(date, category)
            
            # Weekly patterns (weekends different)
            weekly_multiplier = self._get_weekly_multiplier(date)
            
            # Special events (Black Friday, Prime Day, etc.)
            event_multiplier = self._get_event_multiplier(date, category)
            
            # COVID-19 impact (2022 recovery, supply chain issues)
            covid_multiplier = self._get_covid_multiplier(date, category)
            
            # Random noise
            noise = np.random.normal(1.0, volatility * 0.1)
            
            # Calculate final demand
            daily_demand = (
                base_demand * 
                trend_multiplier * 
                seasonal_demand * 
                weekly_multiplier * 
                event_multiplier * 
                covid_multiplier * 
                noise
            )
            
            # Ensure non-negative demand
            daily_demand = max(0, daily_demand)
            
            # Add some zero-demand days (stockouts, product launches, etc.)
            if random.random() < 0.02:  # 2% chance of zero demand
                daily_demand = 0
            
            timeline_data.append({
                'product_id': product_id,
                'ds': date.strftime('%Y-%m-%d'),
                'y': round(daily_demand, 2),
                'category': category
            })
        
        return timeline_data
    
    def _calculate_seasonal_multiplier(self, date: datetime, category: str) -> float:
        """Calculate seasonal demand multiplier based on month and category"""
        
        month = date.month
        
        # Different seasonal patterns by category
        if category == "electronics":
            # Peak in November (Black Friday), December (holidays), back-to-school
            seasonal_map = {
                1: 0.8, 2: 0.7, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.8,
                7: 1.0, 8: 1.2, 9: 1.1, 10: 1.1, 11: 1.5, 12: 1.4
            }
        elif category == "clothing":
            # Spring/summer and fall/winter collections
            seasonal_map = {
                1: 0.8, 2: 1.1, 3: 1.3, 4: 1.2, 5: 1.1, 6: 1.0,
                7: 0.9, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.4, 12: 1.2
            }
        elif category == "home_garden":
            # Spring peak for garden, year-end for home improvement
            seasonal_map = {
                1: 0.9, 2: 1.0, 3: 1.3, 4: 1.4, 5: 1.3, 6: 1.1,
                7: 1.0, 8: 0.9, 9: 1.0, 10: 1.1, 11: 1.2, 12: 1.1
            }
        elif category == "health_beauty":
            # New Year resolutions, summer prep
            seasonal_map = {
                1: 1.3, 2: 1.2, 3: 1.1, 4: 1.2, 5: 1.3, 6: 1.2,
                7: 1.1, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.2
            }
        else:
            # Default seasonal pattern
            seasonal_map = {
                1: 0.9, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.1, 6: 1.1,
                7: 1.0, 8: 1.0, 9: 1.0, 10: 1.1, 11: 1.3, 12: 1.2
            }
        
        return seasonal_map.get(month, 1.0)
    
    def _get_weekly_multiplier(self, date: datetime) -> float:
        """Get weekly demand multiplier (weekends vs weekdays)"""
        
        day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
        
        # E-commerce pattern: higher on weekends and evenings
        weekly_pattern = {
            0: 1.0,  # Monday
            1: 0.9,  # Tuesday  
            2: 0.9,  # Wednesday
            3: 1.0,  # Thursday
            4: 1.1,  # Friday
            5: 1.3,  # Saturday
            6: 1.2   # Sunday
        }
        
        return weekly_pattern.get(day_of_week, 1.0)
    
    def _get_event_multiplier(self, date: datetime, category: str) -> float:
        """Get multiplier for special shopping events"""
        
        month = date.month
        day = date.day
        
        # Black Friday (last Friday of November)
        if month == 11 and 22 <= day <= 28 and date.weekday() == 4:
            return 3.0 if category in ["electronics", "clothing"] else 2.0
        
        # Cyber Monday
        if month == 11 and 25 <= day <= 30 and date.weekday() == 0:
            return 2.5 if category == "electronics" else 1.5
        
        # Prime Day (mid July)
        if month == 7 and 10 <= day <= 16:
            return 2.0
        
        # Valentine's Day
        if month == 2 and day == 14:
            return 1.8 if category in ["clothing", "health_beauty"] else 1.2
        
        # Back to School (late August)
        if month == 8 and day >= 20:
            return 1.6 if category in ["electronics", "clothing"] else 1.1
        
        # Holiday season (December)
        if month == 12 and day >= 15:
            return 1.8
        
        return 1.0
    
    def _get_covid_multiplier(self, date: datetime, category: str) -> float:
        """Get COVID-19 impact multiplier for different periods"""
        
        year = date.year
        month = date.month
        
        # 2022: Recovery and supply chain issues
        if year == 2022:
            if category in ["electronics", "home_garden"]:
                return 1.2  # High demand, supply chain issues
            elif category == "clothing":
                return 0.9  # Still recovering from lockdowns
            else:
                return 1.0
        
        # 2023: Normalization but inflation impact
        elif year == 2023:
            if category in ["health_beauty", "electronics"]:
                return 1.1  # Strong demand
            elif category == "books_media":
                return 0.8  # Digital shift impact
            else:
                return 1.0
        
        # 2024: Post-pandemic normal with new patterns
        else:
            return 1.0
    
    def generate_sample_analysis(self, df: pd.DataFrame) -> dict:
        """Generate sample analysis of the created dataset"""
        
        # Convert ds to datetime if it's not already
        df_analysis = df.copy()
        df_analysis['ds'] = pd.to_datetime(df_analysis['ds'])
        
        analysis = {
            "dataset_summary": {
                "total_records": len(df_analysis),
                "unique_products": df_analysis['product_id'].nunique(),
                "date_range": {
                    "start": str(df_analysis['ds'].min()),
                    "end": str(df_analysis['ds'].max())
                },
                "categories": df_analysis['category'].value_counts().to_dict(),
                "total_demand": float(df_analysis['y'].sum()),
                "average_daily_demand": float(df_analysis['y'].mean())
            },
            "top_products": df_analysis.groupby('product_id')['y'].sum().sort_values(ascending=False).head(10).to_dict(),
            "seasonal_trends": df_analysis.groupby(df_analysis['ds'].dt.to_period('M'))['y'].sum().tail(12).to_dict()
        }
        
        return analysis

# Usage function
def create_modern_training_data():
    """Create modern training data for the seasonal prediction system"""
    
    generator = ModernEcommerceDataGenerator()
    
    # Generate dataset
    df = generator.generate_modern_dataset(
        num_products=2000,  # Generate 2000 modern products
        output_path="ai-services/seasonal-inventory/data/processed/daily_demand_by_product_modern.csv"
    )
    
    # Generate analysis
    analysis = generator.generate_sample_analysis(df)
    
    print("📊 Modern Dataset Analysis:")
    print(f"Total Records: {analysis['dataset_summary']['total_records']:,}")
    print(f"Products: {analysis['dataset_summary']['unique_products']:,}")
    print(f"Date Range: {analysis['dataset_summary']['date_range']['start']} to {analysis['dataset_summary']['date_range']['end']}")
    print(f"Categories: {analysis['dataset_summary']['categories']}")
    print(f"Average Daily Demand: {analysis['dataset_summary']['average_daily_demand']:.2f}")
    
    return df, analysis

if __name__ == "__main__":
    df, analysis = create_modern_training_data()
