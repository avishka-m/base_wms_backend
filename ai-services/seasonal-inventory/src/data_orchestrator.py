"""
Data Orchestrator for Seasonal Inventory Prediction

This module orchestrates data collection from all sources and creates
a unified dataset for Prophet model training.
"""

import pandas as pd
import logging
from typing import Dict
from datetime import datetime
from pathlib import Path
import asyncio
import numpy as np

from data_collection.kaggle_downloader import KaggleDataDownloader
from data_collection.web_scraper import WebDataScraper
from data_collection.wms_data_extractor import WMSDataExtractor
from config import PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeasonalDataOrchestrator:
    """
    Orchestrates data collection from all sources and creates unified datasets
    for seasonal inventory forecasting.
    """
    
    def __init__(self, processed_dir: str = PROCESSED_DIR):
        """
        Initialize the data orchestrator.
        
        Args:
            processed_dir: Directory to store processed datasets
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data collectors
        self.kaggle_downloader = KaggleDataDownloader()
        self.web_scraper = WebDataScraper()
        self.wms_extractor = WMSDataExtractor()
        
        logger.info("🎼 Seasonal Data Orchestrator initialized")
    
    def collect_kaggle_datasets(self, priority: str = "high_priority") -> Dict[str, pd.DataFrame]:
        """
        Collect and process Kaggle datasets.
        
        Args:
            priority: Priority level for datasets
            
        Returns:
            Dictionary of processed DataFrames
        """
        logger.info(f"📦 Collecting {priority} Kaggle datasets")
        
        # Download datasets
        download_results = self.kaggle_downloader.download_priority_datasets(priority)
        
        # Process successful downloads
        processed_datasets = {}
        for dataset_name, success in download_results.items():
            if success:
                try:
                    df = self.kaggle_downloader.process_for_prophet(dataset_name)
                    if df is not None and not df.empty:
                        processed_datasets[dataset_name] = df
                        logger.info(f"✅ Processed {dataset_name}: {len(df)} records")
                    else:
                        logger.warning(f"⚠️ Failed to process {dataset_name}")
                except Exception as e:
                    logger.error(f"❌ Error processing {dataset_name}: {e}")
        
        return processed_datasets
    
    def collect_external_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect external market data via web scraping.
        
        Returns:
            Dictionary of external DataFrames
        """
        logger.info("🌐 Collecting external market data")
        
        try:
            # Get all external data
            external_data = self.web_scraper.scrape_all_external_data()
            
            # Create comprehensive features dataset
            features_df = self.web_scraper.create_external_features_dataset()
            
            if not features_df.empty:
                external_data["comprehensive_features"] = features_df
            
            return external_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting external data: {e}")
            return {}
    
    async def collect_wms_data(self, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> pd.DataFrame:
        """
        Collect WMS historical data.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            WMS DataFrame
        """
        logger.info("📦 Collecting WMS historical data")
        
        try:
            df = await self.wms_extractor.create_comprehensive_dataset(start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"❌ Error collecting WMS data: {e}")
            return pd.DataFrame()
        finally:
            await self.wms_extractor.close()
    
    def merge_datasets(self, 
                      kaggle_datasets: Dict[str, pd.DataFrame],
                      external_data: Dict[str, pd.DataFrame],
                      wms_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all datasets into a unified format for Prophet.
        
        Args:
            kaggle_datasets: Kaggle datasets
            external_data: External market data
            wms_data: WMS historical data
            
        Returns:
            Unified DataFrame ready for Prophet training
        """
        logger.info("🔗 Merging all datasets")
        
        unified_records = []
        
        # Process WMS data (highest priority - real business data)
        if not wms_data.empty:
            logger.info(f"   📊 Adding WMS data: {len(wms_data)} records")
            for _, row in wms_data.iterrows():
                unified_records.append({
                    "ds": row["ds"],
                    "y": row["y"],
                    "product_id": row["product_id"],
                    "category": row.get("category", "unknown"),
                    "warehouse_id": row.get("warehouse_id", "main"),
                    "data_source": "wms",
                    "unit_price": row.get("unit_price", 0),
                    "total_value": row.get("total_value", 0)
                })
        
        # Process Kaggle datasets (for pattern learning)
        for dataset_name, df in kaggle_datasets.items():
            logger.info(f"   📦 Adding Kaggle dataset {dataset_name}: {len(df)} records")
            
            # Sample data if too large (keep recent data)
            if len(df) > 50000:
                df = df.tail(50000)
                logger.info(f"   ✂️ Sampled to {len(df)} most recent records")
            
            for _, row in df.iterrows():
                unified_records.append({
                    "ds": row["ds"],
                    "y": row["y"],
                    "product_id": row.get("product_id", f"kaggle_{len(unified_records)}"),
                    "category": "external_data",
                    "warehouse_id": "external",
                    "data_source": f"kaggle_{dataset_name}",
                    "unit_price": 0,
                    "total_value": 0
                })
        
        # Create unified DataFrame
        if unified_records:
            unified_df = pd.DataFrame(unified_records)
            
            # Data quality checks and cleaning
            unified_df = self._clean_unified_dataset(unified_df)
            
            # Add external features if available
            if "comprehensive_features" in external_data:
                unified_df = self._add_external_features(unified_df, external_data["comprehensive_features"])
            
            logger.info(f"✅ Unified dataset created: {len(unified_df)} records")
            return unified_df
        
        logger.warning("⚠️ No data to merge")
        return pd.DataFrame()
    
    def _clean_unified_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the unified dataset.
        
        Args:
            df: Raw unified DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Cleaning unified dataset")
        
        initial_records = len(df)
        
        # Data type conversion
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        
        # Remove invalid records
        df = df.dropna(subset=["ds", "y"])
        df = df[df["y"] > 0]  # Only positive demand
        
        # Remove outliers (values more than 3 std devs from mean)
        mean_demand = df["y"].mean()
        std_demand = df["y"].std()
        outlier_threshold = mean_demand + (3 * std_demand)
        df = df[df["y"] <= outlier_threshold]
        
        # Sort by date
        df = df.sort_values("ds").reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["ds", "product_id", "y"], keep="last")
        
        final_records = len(df)
        removed_records = initial_records - final_records
        
        logger.info(f"   📊 Cleaning complete: removed {removed_records} invalid records")
        logger.info(f"   📊 Final dataset: {final_records} clean records")
        
        return df
    
    def _add_external_features(self, 
                              demand_df: pd.DataFrame, 
                              features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add external features to the demand dataset.
        
        Args:
            demand_df: Main demand dataset
            features_df: External features dataset
            
        Returns:
            Enhanced DataFrame with external features
        """
        logger.info("🌟 Adding external features")
        
        # Merge on date
        enhanced_df = demand_df.merge(
            features_df, 
            on="ds", 
            how="left"
        )
        
        # Fill missing external features
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in enhanced_df.columns:
                enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
        
        # Fill missing boolean columns
        boolean_columns = ["is_holiday", "is_weekend", "is_month_end", "is_quarter_end"]
        for col in boolean_columns:
            if col in enhanced_df.columns:
                enhanced_df[col] = enhanced_df[col].fillna(False)
        
        feature_count = len(features_df.columns) - 1  # Exclude 'ds' column
        logger.info(f"   ✅ Added {feature_count} external features")
        
        return enhanced_df
    
    def create_product_specific_datasets(self, unified_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create product-specific datasets for individual forecasting.
        
        Args:
            unified_df: Unified dataset
            
        Returns:
            Dictionary mapping product IDs to their specific datasets
        """
        logger.info("🏷️ Creating product-specific datasets")
        
        product_datasets = {}
        
        # Group by product and aggregate daily demand
        product_groups = unified_df.groupby("product_id")
        
        for product_id, group in product_groups:
            # Aggregate daily demand for each product
            daily_demand = group.groupby("ds").agg({
                "y": "sum",  # Total daily demand
                "category": "first",
                "warehouse_id": "first",
                "unit_price": "mean",
                "total_value": "sum"
            }).reset_index()
            
            # Only keep products with sufficient history (at least 30 days)
            if len(daily_demand) >= 30:
                product_datasets[product_id] = daily_demand
                
                # Save individual product dataset
                output_file = self.processed_dir / f"product_{product_id}_demand.csv"
                daily_demand.to_csv(output_file, index=False)
        
        logger.info(f"📊 Created {len(product_datasets)} product-specific datasets")
        return product_datasets
    
    async def orchestrate_full_data_collection(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data collection and processing pipeline.
        
        Returns:
            Dictionary containing all processed datasets
        """
        logger.info("🚀 Starting comprehensive data collection orchestration")
        
        results = {}
        
        # Step 1: Collect Kaggle datasets
        logger.info("\n📦 Step 1: Collecting Kaggle datasets")
        kaggle_datasets = self.collect_kaggle_datasets("high_priority")
        results["kaggle"] = kaggle_datasets
        
        # Step 2: Collect external data
        logger.info("\n🌐 Step 2: Collecting external market data")
        external_data = self.collect_external_data()
        results["external"] = external_data
        
        # Step 3: Collect WMS data
        logger.info("\n📦 Step 3: Collecting WMS historical data")
        wms_data = await self.collect_wms_data()
        results["wms"] = wms_data
        
        # Step 4: Merge all datasets
        logger.info("\n🔗 Step 4: Merging all datasets")
        unified_df = self.merge_datasets(kaggle_datasets, external_data, wms_data)
        results["unified"] = unified_df
        
        # Step 5: Create product-specific datasets
        if not unified_df.empty:
            logger.info("\n🏷️ Step 5: Creating product-specific datasets")
            product_datasets = self.create_product_specific_datasets(unified_df)
            results["products"] = product_datasets
            
            # Save unified dataset
            unified_file = self.processed_dir / "unified_seasonal_dataset.csv"
            unified_df.to_csv(unified_file, index=False)
            logger.info(f"💾 Saved unified dataset to {unified_file}")
        
        # Step 6: Generate summary report
        logger.info("\n📋 Step 6: Generating collection summary")
        summary = self._generate_collection_summary(results)
        results["summary"] = summary
        
        logger.info("✅ Data collection orchestration complete!")
        return results
    
    def _generate_collection_summary(self, results: Dict) -> Dict:
        """
        Generate a summary report of data collection results.
        
        Args:
            results: Collection results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "collection_timestamp": datetime.now().isoformat(),
            "kaggle_datasets": {},
            "external_data": {},
            "wms_data": {},
            "unified_dataset": {},
            "product_datasets": 0
        }
        
        # Kaggle summary
        if "kaggle" in results:
            for name, df in results["kaggle"].items():
                summary["kaggle_datasets"][name] = {
                    "records": len(df),
                    "date_range": f"{df['ds'].min()} to {df['ds'].max()}" if not df.empty else "No data"
                }
        
        # External data summary
        if "external" in results:
            for name, df in results["external"].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    summary["external_data"][name] = {
                        "records": len(df),
                        "features": len(df.columns)
                    }
        
        # WMS data summary
        if "wms" in results and isinstance(results["wms"], pd.DataFrame) and not results["wms"].empty:
            df = results["wms"]
            summary["wms_data"] = {
                "records": len(df),
                "unique_products": df["product_id"].nunique(),
                "date_range": f"{df['ds'].min()} to {df['ds'].max()}"
            }
        
        # Unified dataset summary
        if "unified" in results and isinstance(results["unified"], pd.DataFrame) and not results["unified"].empty:
            df = results["unified"]
            summary["unified_dataset"] = {
                "total_records": len(df),
                "unique_products": df["product_id"].nunique(),
                "data_sources": df["data_source"].unique().tolist(),
                "date_range": f"{df['ds'].min()} to {df['ds'].max()}",
                "total_demand": float(df["y"].sum()),
                "features": len(df.columns)
            }
        
        # Product datasets summary
        if "products" in results:
            summary["product_datasets"] = len(results["products"])
        
        return summary


async def main():
    """Main function to demonstrate the data orchestration process."""
    print("🎼 Seasonal Inventory Data Orchestrator")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = SeasonalDataOrchestrator()
    
    # Run full data collection
    results = await orchestrator.orchestrate_full_data_collection()
    
    # Display summary
    if "summary" in results:
        summary = results["summary"]
        
        print("\n📊 Data Collection Summary:")
        print("=" * 40)
        
        print(f"🕐 Collection completed: {summary['collection_timestamp']}")
        
        if summary["kaggle_datasets"]:
            print(f"\n📦 Kaggle Datasets: {len(summary['kaggle_datasets'])}")
            for name, info in summary["kaggle_datasets"].items():
                print(f"   • {name}: {info['records']} records")
        
        if summary["external_data"]:
            print(f"\n🌐 External Data Sources: {len(summary['external_data'])}")
            for name, info in summary["external_data"].items():
                print(f"   • {name}: {info['records']} records, {info['features']} features")
        
        if summary["wms_data"]:
            print(f"\n📦 WMS Data: {summary['wms_data']['records']} records")
            print(f"   • Products: {summary['wms_data']['unique_products']}")
            print(f"   • Date range: {summary['wms_data']['date_range']}")
        
        if summary["unified_dataset"]:
            unified = summary["unified_dataset"]
            print("\n🔗 Unified Dataset:")
            print(f"   • Total records: {unified['total_records']:,}")
            print(f"   • Unique products: {unified['unique_products']}")
            print(f"   • Data sources: {len(unified['data_sources'])}")
            print(f"   • Total demand: {unified['total_demand']:,.0f}")
            print(f"   • Features: {unified['features']}")
        
        print(f"\n🏷️ Product-specific datasets: {summary['product_datasets']}")
        
        print("\n✅ Data collection orchestration completed successfully!")
    
    else:
        print("⚠️ No summary data available. Check the collection process.")


if __name__ == "__main__":
    asyncio.run(main())
