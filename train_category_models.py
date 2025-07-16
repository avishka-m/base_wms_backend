#!/usr/bin/env python3
"""
🎯 Category Model Training Script - Train 6 Prophet Models for 6 Categories
================================================================================

This script trains optimized Prophet models for each of the 6 product categories:
- books_media
- clothing  
- electronics
- health_beauty
- home_garden
- sports_outdoors

Each category uses optimized hyperparameters for best forecasting performance.

Usage:
    python train_category_models.py [--verbose] [--force-retrain]

Options:
    --verbose        Show detailed training progress
    --force-retrain  Retrain models even if they already exist
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent
sys.path.insert(0, str(project_root))

# Import the Prophet forecasting service
try:
    from app.services.prophet_forecasting_service import ProphetForecastingService
    from app.services.prophet_forecasting_service import get_prophet_forecasting_service
except ImportError as e:
    print(f"Error importing forecasting service: {e}")
    print("Make sure you're running from the base_wms_backend directory")
    sys.exit(1)

# Configure logging
def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'category_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def train_category_models(force_retrain=False, verbose=False):
    """🚀 Main function to train all 6 category models"""
    
    print("=" * 80)
    print("🎯 CATEGORY MODEL TRAINING - 6 Prophet Models")
    print("=" * 80)
    print()
    
    try:
        # Initialize the service
        print("📊 Initializing Prophet Forecasting Service...")
        service = get_prophet_forecasting_service()
        
        # Check service status
        status = service.get_service_status()
        if status["status"] != "available":
            print(f"❌ Service unavailable: {status.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Service ready. Models path: {status['models_path']}")
        print()
        
        # Check current category status
        print("📋 Checking current category model status...")
        category_status = service.get_category_status()
        
        if category_status["status"] == "ready":
            print(f"📦 Total categories: {category_status['summary']['total_categories']}")
            print(f"✅ Trained models: {category_status['summary']['trained_categories']}")
            print(f"⏳ Untrained models: {category_status['summary']['untrained_categories']}")
            print(f"📈 Training completion: {category_status['summary']['training_completion']}")
            print()
            
            # Show category distribution
            print("📊 Product distribution by category:")
            for category, count in category_status['product_distribution'].items():
                status_icon = "✅" if category_status['category_models'][category]['status'] == "trained" else "⏳"
                print(f"   {status_icon} {category}: {count} products")
            print()
            
            # Check if we need to train
            if not force_retrain and category_status['summary']['trained_categories'] == category_status['summary']['total_categories']:
                print("🎉 All category models are already trained!")
                print("   Use --force-retrain to retrain existing models")
                return True
        
        # Start training
        print("🚀 Starting category model training...")
        print("   This may take several minutes depending on data size...")
        print()
        
        start_time = datetime.now()
        
        # Train all category models
        training_result = await service.train_category_models()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print("=" * 80)
        print("📊 TRAINING RESULTS")
        print("=" * 80)
        
        if training_result["status"] == "success":
            print(f"🎉 Training completed successfully!")
            print(f"⏱️  Total training time: {training_duration}")
            print()
            
            # Show detailed results
            print("📋 Category Training Results:")
            print()
            
            for category, result in training_result["category_results"].items():
                if result["status"] == "success":
                    print(f"✅ {category.upper()}")
                    print(f"   📁 Model: {Path(result['model_path']).name}")
                    print(f"   📊 Data points: {result['data_info']['total_records']}")
                    print(f"   🏷️  Products: {result['data_info']['products_in_category']}")
                    print(f"   📈 Avg daily demand: {result['data_info']['avg_daily_demand']}")
                    print(f"   🔥 Peak daily demand: {result['data_info']['peak_daily_demand']}")
                    print(f"   📅 Date range: {result['data_info']['date_range']}")
                    print()
                elif result["status"] == "error":
                    print(f"❌ {category.upper()}")
                    print(f"   Error: {result['message']}")
                    print()
                elif result["status"] == "skipped":
                    print(f"⏭️  {category.upper()}")
                    print(f"   Reason: {result['message']}")
                    print()
            
            # Summary
            summary = training_result["summary"]
            print("📈 TRAINING SUMMARY")
            print(f"   Total categories: {summary['total_categories']}")
            print(f"   ✅ Successful: {summary['successful']}")
            print(f"   ❌ Failed: {summary['failed']}")
            print(f"   📊 Success rate: {summary['success_rate']}")
            print()
            
            if summary['successful'] == summary['total_categories']:
                print("🎉 ALL CATEGORY MODELS TRAINED SUCCESSFULLY!")
                print("   You can now use category-based forecasting for better accuracy")
                
                # Show next steps
                print()
                print("🚀 NEXT STEPS:")
                print("   1. Test the models with: python test_category_forecasts.py")
                print("   2. Use the API endpoints for category-based predictions")
                print("   3. Monitor model performance and retrain as needed")
                
                return True
            else:
                print("⚠️  Some category models failed to train")
                print("   Check the logs for detailed error information")
                return False
                
        else:
            print(f"❌ Training failed: {training_result['message']}")
            return False
            
    except Exception as e:
        print(f"💥 Unexpected error during training: {e}")
        logging.error(f"Training error: {e}", exc_info=True)
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train 6 Prophet models for product categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed training progress'
    )
    
    parser.add_argument(
        '--force-retrain', '-f',
        action='store_true',
        help='Retrain models even if they already exist'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run the training
    try:
        success = asyncio.run(train_category_models(
            force_retrain=args.force_retrain,
            verbose=args.verbose
        ))
        
        if success:
            print()
            print("🎯 Category model training completed successfully!")
            sys.exit(0)
        else:
            print()
            print("❌ Category model training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
