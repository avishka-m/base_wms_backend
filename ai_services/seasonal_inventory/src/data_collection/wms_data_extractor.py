"""
WMS Data Extractor for Historical Inventory Analysis

This module extracts historical warehouse data from the WMS database
to build comprehensive datasets for seasonal forecasting.
"""

import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import requests

from config import (
    MONGODB_URL, DATABASE_NAME, WMS_API_BASE_URL, WMS_API_TIMEOUT,
    PROCESSED_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WMSDataExtractor:
    """
    Extracts historical data from the WMS system for seasonal analysis.
    """
    
    def __init__(self, 
                 mongodb_url: str = MONGODB_URL,
                 database_name: str = DATABASE_NAME,
                 api_base_url: str = WMS_API_BASE_URL):
        """
        Initialize the WMS data extractor.
        
        Args:
            mongodb_url: MongoDB connection URL
            database_name: Database name
            api_base_url: WMS API base URL
        """
        self.mongodb_url = mongodb_url
        self.database_name = database_name
        self.api_base_url = api_base_url
        
        # Setup MongoDB client
        self.mongo_client = AsyncIOMotorClient(mongodb_url)
        self.db = self.mongo_client[database_name]
        
        # Setup API session
        self.session = requests.Session()
        self.session.timeout = WMS_API_TIMEOUT
        
        logger.info("📦 WMS Data Extractor initialized")
    
    async def extract_inventory_transactions(self, 
                                           start_date: datetime = None,
                                           end_date: datetime = None,
                                           product_ids: List[str] = None) -> pd.DataFrame:
        """
        Extract historical inventory transactions from MongoDB.
        
        Args:
            start_date: Start date for data extraction
            end_date: End date for data extraction
            product_ids: List of specific product IDs to extract
            
        Returns:
            DataFrame with inventory transaction data
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365*2)
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"📊 Extracting inventory transactions from {start_date} to {end_date}")
            
            # Build MongoDB query
            query = {
                "created_at": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            if product_ids:
                query["product_id"] = {"$in": product_ids}
            
            # Extract from various collections
            collections_to_extract = [
                "inventory_transactions",
                "stock_movements", 
                "receiving_records",
                "picking_records",
                "shipping_records"
            ]
            
            all_transactions = []
            
            for collection_name in collections_to_extract:
                try:
                    collection = self.db[collection_name]
                    
                    # Get documents
                    cursor = collection.find(query)
                    documents = await cursor.to_list(length=None)
                    
                    logger.info(f"   📄 {collection_name}: {len(documents)} records")
                    
                    for doc in documents:
                        # Standardize the document structure
                        transaction = {
                            "ds": doc.get("created_at", doc.get("timestamp", doc.get("date"))),
                            "product_id": doc.get("product_id", doc.get("sku", doc.get("item_id"))),
                            "quantity": doc.get("quantity", doc.get("qty", 0)),
                            "transaction_type": collection_name,
                            "warehouse_id": doc.get("warehouse_id", doc.get("location_id")),
                            "unit_price": doc.get("unit_price", doc.get("price", 0)),
                            "total_value": doc.get("total_value", 0),
                            "customer_id": doc.get("customer_id"),
                            "supplier_id": doc.get("supplier_id"),
                            "order_id": doc.get("order_id"),
                            "category": doc.get("category", doc.get("product_category")),
                            "brand": doc.get("brand", doc.get("product_brand"))
                        }
                        
                        # Calculate total_value if missing
                        if not transaction["total_value"] and transaction["quantity"] and transaction["unit_price"]:
                            transaction["total_value"] = transaction["quantity"] * transaction["unit_price"]
                        
                        all_transactions.append(transaction)
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract from {collection_name}: {e}")
                    continue
            
            if all_transactions:
                df = pd.DataFrame(all_transactions)
                
                # Data cleaning and standardization
                df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
                df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
                df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
                
                # Remove invalid records
                df = df.dropna(subset=["ds", "product_id"])
                df = df[df["quantity"] > 0]  # Only positive quantities
                
                # Sort by date
                df = df.sort_values("ds").reset_index(drop=True)
                
                logger.info(f"✅ Extracted {len(df)} valid inventory transactions")
                
                return df
            
            logger.warning("⚠️ No inventory transactions found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to extract inventory transactions: {e}")
            return pd.DataFrame()
    
    async def extract_sales_data(self, 
                               start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
        """
        Extract sales data for demand analysis.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with sales data
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365*2)
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"💰 Extracting sales data from {start_date} to {end_date}")
            
            query = {
                "order_date": {
                    "$gte": start_date,
                    "$lte": end_date
                },
                "status": {"$in": ["completed", "shipped", "delivered"]}
            }
            
            # Extract from orders and order_items collections
            orders_collection = self.db["orders"]
            order_items_collection = self.db["order_items"]
            
            # Get completed orders
            orders_cursor = orders_collection.find(query)
            orders = await orders_cursor.to_list(length=None)
            
            sales_data = []
            
            for order in orders:
                order_id = str(order["_id"])
                
                # Get order items
                items_cursor = order_items_collection.find({"order_id": order_id})
                items = await items_cursor.to_list(length=None)
                
                for item in items:
                    sales_record = {
                        "ds": order.get("order_date"),
                        "order_id": order_id,
                        "product_id": item.get("product_id"),
                        "quantity": item.get("quantity", 0),
                        "unit_price": item.get("unit_price", 0),
                        "total_value": item.get("total_price", 0),
                        "customer_id": order.get("customer_id"),
                        "warehouse_id": order.get("warehouse_id"),
                        "shipping_method": order.get("shipping_method"),
                        "payment_method": order.get("payment_method"),
                        "order_status": order.get("status"),
                        "category": item.get("category"),
                        "discount": item.get("discount", 0)
                    }
                    sales_data.append(sales_record)
            
            if sales_data:
                df = pd.DataFrame(sales_data)
                
                # Data cleaning
                df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
                df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
                df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
                
                # Remove invalid records
                df = df.dropna(subset=["ds", "product_id"])
                df = df[df["quantity"] > 0]
                
                # Sort by date
                df = df.sort_values("ds").reset_index(drop=True)
                
                logger.info(f"✅ Extracted {len(df)} sales records")
                
                return df
            
            logger.warning("⚠️ No sales data found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to extract sales data: {e}")
            return pd.DataFrame()
    
    async def extract_stock_levels(self, 
                                 start_date: datetime = None,
                                 end_date: datetime = None) -> pd.DataFrame:
        """
        Extract historical stock level data.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            DataFrame with stock level history
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365*2)
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"📊 Extracting stock levels from {start_date} to {end_date}")
            
            query = {
                "updated_at": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            collection = self.db["inventory_levels"]
            cursor = collection.find(query)
            documents = await cursor.to_list(length=None)
            
            stock_data = []
            
            for doc in documents:
                stock_record = {
                    "ds": doc.get("updated_at"),
                    "product_id": doc.get("product_id"),
                    "warehouse_id": doc.get("warehouse_id"),
                    "current_stock": doc.get("current_stock", 0),
                    "reserved_stock": doc.get("reserved_stock", 0),
                    "available_stock": doc.get("available_stock", 0),
                    "reorder_point": doc.get("reorder_point", 0),
                    "max_stock": doc.get("max_stock", 0),
                    "min_stock": doc.get("min_stock", 0),
                    "unit_cost": doc.get("unit_cost", 0),
                    "category": doc.get("category"),
                    "supplier_id": doc.get("supplier_id"),
                    "lead_time_days": doc.get("lead_time_days", 0)
                }
                stock_data.append(stock_record)
            
            if stock_data:
                df = pd.DataFrame(stock_data)
                
                # Data cleaning
                df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
                df = df.dropna(subset=["ds", "product_id"])
                
                # Sort by date
                df = df.sort_values("ds").reset_index(drop=True)
                
                logger.info(f"✅ Extracted {len(df)} stock level records")
                
                return df
            
            logger.warning("⚠️ No stock level data found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to extract stock levels: {e}")
            return pd.DataFrame()
    
    def extract_via_api(self, endpoint: str, params: Dict = None) -> pd.DataFrame:
        """
        Extract data via WMS REST API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            DataFrame with API response data
        """
        try:
            url = f"{self.api_base_url}/{endpoint}"
            
            logger.info(f"🌐 Extracting data from API: {endpoint}")
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                logger.warning(f"⚠️ Unexpected API response format from {endpoint}")
                return pd.DataFrame()
            
            logger.info(f"✅ Extracted {len(df)} records from API")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to extract from API {endpoint}: {e}")
            return pd.DataFrame()
    
    async def create_comprehensive_dataset(self, 
                                         start_date: datetime = None,
                                         end_date: datetime = None) -> pd.DataFrame:
        """
        Create a comprehensive dataset combining all WMS data sources.
        
        Args:
            start_date: Start date for extraction
            end_date: End date for extraction
            
        Returns:
            Combined DataFrame with all WMS data
        """
        logger.info("🔄 Creating comprehensive WMS dataset")
        
        # Extract all data sources
        tasks = [
            self.extract_inventory_transactions(start_date, end_date),
            self.extract_sales_data(start_date, end_date),
            self.extract_stock_levels(start_date, end_date)        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        inventory_df = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
        sales_df = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
        # stock_df = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()  # Reserved for future use
        
        # Combine all datasets
        combined_data = []
        
        # Process inventory transactions
        if not inventory_df.empty:
            for _, row in inventory_df.iterrows():
                combined_data.append({
                    "ds": row["ds"],
                    "product_id": row["product_id"],
                    "y": abs(row["quantity"]),  # Use absolute quantity as demand proxy
                    "data_source": "inventory_transaction",
                    "transaction_type": row["transaction_type"],
                    "warehouse_id": row["warehouse_id"],
                    "category": row["category"],
                    "unit_price": row["unit_price"],
                    "total_value": row["total_value"]
                })
        
        # Process sales data (primary demand signal)
        if not sales_df.empty:
            for _, row in sales_df.iterrows():
                combined_data.append({
                    "ds": row["ds"],
                    "product_id": row["product_id"],
                    "y": row["quantity"],  # Direct demand
                    "data_source": "sales",
                    "transaction_type": "sale",
                    "warehouse_id": row["warehouse_id"],
                    "category": row["category"],
                    "unit_price": row["unit_price"],
                    "total_value": row["total_value"]
                })
        
        if combined_data:
            df = pd.DataFrame(combined_data)
            
            # Aggregate by date and product for Prophet format
            prophet_df = df.groupby(["ds", "product_id"]).agg({
                "y": "sum",  # Total demand
                "category": "first",
                "warehouse_id": "first",
                "unit_price": "mean",
                "total_value": "sum"
            }).reset_index()
            
            # Sort by date
            prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
            
            logger.info(f"📊 Comprehensive dataset created with {len(prophet_df)} records")
            
            # Save the dataset
            output_dir = Path(PROCESSED_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / "wms_historical_data.csv"
            prophet_df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved WMS dataset to {output_file}")
            
            return prophet_df
        
        logger.warning("⚠️ No data found to create comprehensive dataset")
        return pd.DataFrame()
    
    async def close(self):
        """Close database connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if hasattr(self.session, 'close'):
            self.session.close()


async def main():
    """Main function to demonstrate WMS data extraction."""
    print("📦 WMS Data Extractor for Seasonal Inventory Prediction")
    print("=" * 60)
    
    # Initialize extractor
    extractor = WMSDataExtractor()
    
    try:
        # Extract comprehensive dataset
        print("\n🔄 Extracting comprehensive WMS dataset...")
        df = await extractor.create_comprehensive_dataset()
        
        if not df.empty:
            print("\n📊 WMS Dataset Summary:")
            print(f"   Total records: {len(df):,}")
            print(f"   Date range: {df['ds'].min()} to {df['ds'].max()}")
            print(f"   Unique products: {df['product_id'].nunique()}")
            print(f"   Total demand: {df['y'].sum():,.0f}")
            print(f"   Categories: {df['category'].unique()}")
            
            # Show sample data
            print("\n👀 Sample WMS data:")
            print(df.head().to_string())
            
        else:
            print("⚠️ No WMS data found. Check database connection and data availability.")
            
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        
    finally:
        # Clean up
        await extractor.close()


if __name__ == "__main__":
    asyncio.run(main())
