"""
Kaggle Dataset Downloader for Seasonal Inventory Prediction

This module handles automatic downloading and processing of Kaggle datasets
for training seasonal forecasting models.
"""

import os
import logging
import pandas as pd
import zipfile
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import hashlib
import json
from datetime import datetime

from config import KAGGLE_DATASETS, DATASETS_DIR, PROCESSED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleDataDownloader:
    """
    Handles downloading and initial processing of Kaggle datasets
    for seasonal inventory forecasting.
    """
    
    def __init__(self, download_dir: str = DATASETS_DIR):
        """
        Initialize the Kaggle downloader.
        
        Args:
            download_dir: Directory to store downloaded datasets
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kaggle API
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            logger.info("✅ Kaggle API authentication successful")
        except Exception as e:
            logger.error(f"❌ Kaggle API authentication failed: {e}")
            raise
            
        # Track downloaded datasets
        self.manifest_file = self.download_dir / "download_manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load the download manifest file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return {"downloads": {}, "last_updated": None}
    
    def _save_manifest(self):
        """Save the download manifest file."""
        self.manifest["last_updated"] = datetime.now().isoformat()
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a Kaggle dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'username/dataset-name')
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            # Get dataset metadata
            dataset = self.api.dataset_view(dataset_name)
            
            # Get file list
            files = self.api.dataset_list_files(dataset_name).files
            
            info = {
                "name": dataset_name,
                "title": dataset.title,
                "description": dataset.description,
                "size": dataset.totalBytes,
                "download_count": dataset.downloadCount,
                "last_updated": dataset.lastUpdated,
                "files": [{"name": f.name, "size": f.totalBytes} for f in files],
                "total_files": len(files)
            }
            
            logger.info(f"📊 Dataset info for {dataset_name}:")
            logger.info(f"   Title: {info['title']}")
            logger.info(f"   Size: {info['size']:,} bytes")
            logger.info(f"   Files: {info['total_files']}")
            
            return info
            
        except Exception as e:
            logger.error(f"❌ Failed to get dataset info for {dataset_name}: {e}")
            return {}
    
    def download_dataset(self, 
                        dataset_name: str, 
                        force_download: bool = False,
                        unzip: bool = True) -> bool:
        """
        Download a specific Kaggle dataset.
        
        Args:
            dataset_name: Name of the dataset to download
            force_download: Whether to force re-download if already exists
            unzip: Whether to unzip downloaded files
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create dataset-specific directory
            dataset_dir = self.download_dir / dataset_name.replace('/', '_')
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already downloaded
            if not force_download and dataset_name in self.manifest["downloads"]:
                logger.info(f"⏭️  Dataset {dataset_name} already downloaded")
                return True
            
            logger.info(f"⬇️  Downloading dataset: {dataset_name}")
            
            # Get dataset info
            info = self.get_dataset_info(dataset_name)
            if not info:
                return False
            
            # Download with progress bar
            with tqdm(total=info.get('size', 0), 
                     unit='B', 
                     unit_scale=True, 
                     desc=f"Downloading {dataset_name}") as pbar:
                
                def progress_callback(current, total):
                    pbar.update(current - pbar.n)
                
                # Download to temporary directory first
                temp_dir = dataset_dir / "temp"
                temp_dir.mkdir(exist_ok=True)
                
                self.api.dataset_download_files(
                    dataset_name, 
                    path=str(temp_dir),
                    unzip=unzip
                )
                
                # Move files to final location
                for temp_file in temp_dir.iterdir():
                    final_path = dataset_dir / temp_file.name
                    if final_path.exists():
                        final_path.unlink()
                    shutil.move(str(temp_file), str(final_path))
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
            
            # Update manifest
            self.manifest["downloads"][dataset_name] = {
                "download_date": datetime.now().isoformat(),
                "info": info,
                "local_path": str(dataset_dir),
                "files": [f.name for f in dataset_dir.iterdir() if f.is_file()]
            }
            self._save_manifest()
            
            logger.info(f"✅ Successfully downloaded {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name}: {e}")
            return False
    
    def download_priority_datasets(self, priority: str = "high_priority") -> Dict[str, bool]:
        """
        Download all datasets in a priority category.
        
        Args:
            priority: Priority level ('high_priority' or 'medium_priority')
            
        Returns:
            Dictionary mapping dataset names to download success status
        """
        datasets = KAGGLE_DATASETS.get(priority, [])
        results = {}
        
        logger.info(f"📦 Downloading {priority} datasets ({len(datasets)} total)")
        
        for i, dataset_config in enumerate(datasets, 1):
            dataset_name = dataset_config["name"]
            logger.info(f"🔄 [{i}/{len(datasets)}] Processing {dataset_name}")
            
            success = self.download_dataset(dataset_name)
            results[dataset_name] = success
            
            if success:
                logger.info(f"✅ [{i}/{len(datasets)}] {dataset_name} completed")
            else:
                logger.error(f"❌ [{i}/{len(datasets)}] {dataset_name} failed")
        
        # Summary
        successful = sum(results.values())
        logger.info(f"📋 Download Summary: {successful}/{len(datasets)} successful")
        
        return results
    
    def preview_dataset(self, dataset_name: str, max_rows: int = 5) -> Optional[pd.DataFrame]:
        """
        Load and preview a downloaded dataset.
        
        Args:
            dataset_name: Name of the dataset
            max_rows: Maximum number of rows to display
            
        Returns:
            DataFrame with preview data or None if failed
        """
        try:
            if dataset_name not in self.manifest["downloads"]:
                logger.error(f"Dataset {dataset_name} not found in downloads")
                return None
            
            dataset_info = self.manifest["downloads"][dataset_name]
            dataset_dir = Path(dataset_info["local_path"])
            
            # Find CSV files
            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {dataset_name}")
                return None
            
            # Load the first CSV file
            main_file = csv_files[0]
            logger.info(f"📖 Previewing {main_file.name} from {dataset_name}")
            
            df = pd.read_csv(main_file, nrows=max_rows * 2)  # Load a bit more for analysis
            
            # Display info
            logger.info(f"📊 Dataset shape: {df.shape}")
            logger.info(f"📊 Columns: {list(df.columns)}")
            logger.info(f"📊 Data types:\n{df.dtypes}")
            
            # Display preview
            print(f"\n📋 Preview of {dataset_name}:")
            print("=" * 80)
            print(df.head(max_rows).to_string())
            print("=" * 80)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to preview {dataset_name}: {e}")
            return None
    
    def process_for_prophet(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Process a downloaded dataset for Prophet format.
        
        Args:
            dataset_name: Name of the dataset to process
            
        Returns:
            Processed DataFrame in Prophet format (ds, y columns) or None
        """
        try:
            if dataset_name not in self.manifest["downloads"]:
                logger.error(f"Dataset {dataset_name} not found")
                return None
            
            # Find dataset configuration
            dataset_config = None
            for priority in ["high_priority", "medium_priority"]:
                for config in KAGGLE_DATASETS.get(priority, []):
                    if config["name"] == dataset_name:
                        dataset_config = config
                        break
                if dataset_config:
                    break
            
            if not dataset_config:
                logger.error(f"Configuration not found for {dataset_name}")
                return None
            
            # Load the dataset
            dataset_info = self.manifest["downloads"][dataset_name]
            dataset_dir = Path(dataset_info["local_path"])
            
            # Find the target file
            target_file = dataset_config.get("target_file", "data.csv")
            file_path = dataset_dir / target_file
            
            if not file_path.exists():
                # Try to find any CSV file
                csv_files = list(dataset_dir.glob("*.csv"))
                if csv_files:
                    file_path = csv_files[0]
                    logger.warning(f"Target file not found, using {file_path.name}")
                else:
                    logger.error(f"No CSV files found in {dataset_name}")
                    return None
            
            logger.info(f"🔄 Processing {file_path.name} for Prophet format")
            
            # Load the data
            df = pd.read_csv(file_path)
            
            # Extract column mappings from config
            date_col = dataset_config.get("date_column")
            quantity_col = dataset_config.get("quantity_column")
            product_col = dataset_config.get("product_column")
            
            # Validate required columns exist
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found in dataset")
                return None
            
            if quantity_col not in df.columns:
                logger.error(f"Quantity column '{quantity_col}' not found in dataset")
                return None
            
            # Create Prophet format DataFrame
            prophet_df = pd.DataFrame()
            
            # Process date column
            prophet_df['ds'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Process target variable (quantity/demand)
            prophet_df['y'] = pd.to_numeric(df[quantity_col], errors='coerce')
            
            # Add product information if available
            if product_col and product_col in df.columns:
                prophet_df['product_id'] = df[product_col].astype(str)
            
            # Remove invalid rows
            initial_rows = len(prophet_df)
            prophet_df = prophet_df.dropna(subset=['ds', 'y'])
            final_rows = len(prophet_df)
            
            if initial_rows > final_rows:
                logger.warning(f"Removed {initial_rows - final_rows} invalid rows")
            
            # Sort by date
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
            # Basic statistics
            logger.info(f"📊 Processed dataset statistics:")
            logger.info(f"   Total records: {len(prophet_df):,}")
            logger.info(f"   Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
            logger.info(f"   Mean demand: {prophet_df['y'].mean():.2f}")
            logger.info(f"   Std demand: {prophet_df['y'].std():.2f}")
            
            # Save processed data
            processed_dir = Path(PROCESSED_DIR)
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = processed_dir / f"{dataset_name.replace('/', '_')}_prophet.csv"
            prophet_df.to_csv(output_file, index=False)
            logger.info(f"💾 Saved processed data to {output_file}")
            
            return prophet_df
            
        except Exception as e:
            logger.error(f"❌ Failed to process {dataset_name}: {e}")
            return None
    
    def get_download_summary(self) -> Dict:
        """Get summary of all downloaded datasets."""
        total_datasets = len(self.manifest["downloads"])
        total_size = sum(info.get("info", {}).get("size", 0) 
                        for info in self.manifest["downloads"].values())
        
        summary = {
            "total_datasets": total_datasets,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "datasets": list(self.manifest["downloads"].keys()),
            "last_updated": self.manifest.get("last_updated")
        }
        
        return summary


def main():
    """Main function to demonstrate usage."""
    print("🚀 Kaggle Dataset Downloader for Seasonal Inventory Prediction")
    print("=" * 70)
    
    # Initialize downloader
    downloader = KaggleDataDownloader()
    
    # Download high priority datasets
    print("\n📦 Downloading high priority datasets...")
    results = downloader.download_priority_datasets("high_priority")
    
    # Preview first successful dataset
    successful_datasets = [name for name, success in results.items() if success]
    if successful_datasets:
        print(f"\n👀 Previewing first dataset: {successful_datasets[0]}")
        downloader.preview_dataset(successful_datasets[0])
        
        print(f"\n🔄 Processing for Prophet format...")
        prophet_df = downloader.process_for_prophet(successful_datasets[0])
        
        if prophet_df is not None:
            print("\n✅ Prophet format sample:")
            print(prophet_df.head().to_string())
    
    # Show summary
    print("\n📋 Download Summary:")
    summary = downloader.get_download_summary()
    print(f"   Total datasets: {summary['total_datasets']}")
    print(f"   Total size: {summary['total_size_mb']:.1f} MB")
    print(f"   Datasets: {', '.join(summary['datasets'])}")


if __name__ == "__main__":
    main()
