#!/usr/bin/env python3
"""
Seasonal Inventory Prediction - Quick Start Script

This script demonstrates how to get started with the seasonal inventory
prediction system. It walks through the basic setup and data collection process.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from data_collection.kaggle_downloader import KaggleDataDownloader
    from data_collection.web_scraper import WebDataScraper
    from data_collection.wms_data_extractor import WMSDataExtractor
    from data_orchestrator import SeasonalDataOrchestrator
    from config import KAGGLE_USERNAME, KAGGLE_KEY
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the seasonal-inventory directory")
    print("And that all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def print_banner():
    """Print welcome banner."""
    print("🚀 SEASONAL INVENTORY PREDICTION - QUICK START")
    print("=" * 60)
    print("This script will help you get started with seasonal inventory forecasting")
    print("using Facebook Prophet and multi-source data integration.\n")

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    issues = []
    
    # Check Kaggle credentials
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        issues.append("Kaggle API credentials not configured")
        print("   ❌ Kaggle API credentials missing")
        print("      Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("      Or run: kaggle config set username YOUR_USERNAME")
        print("              kaggle config set key YOUR_API_KEY")
    else:
        print("   ✅ Kaggle API credentials configured")
    
    # Check if data directories exist
    data_dir = current_dir / "data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("   ✅ Created data directory")
    else:
        print("   ✅ Data directory exists")
      # Check Python dependencies
    try:
        import importlib.util
        
        # Test core dependencies
        deps_available = True
        for dep in ['pandas', 'numpy', 'requests']:
            if importlib.util.find_spec(dep) is None:
                issues.append(f"Missing Python dependency: {dep}")
                print(f"   ❌ Missing dependency: {dep}")
                deps_available = False
        
        if deps_available:
            print("   ✅ Core dependencies available")
    except ImportError as e:
        issues.append(f"Missing Python dependency: {e}")
        print(f"   ❌ Missing dependency: {e}")
    
    try:
        import importlib.util
        if importlib.util.find_spec('kaggle') is not None:
            print("   ✅ Kaggle package available")
        else:
            issues.append("Kaggle package not installed")
            print("   ❌ Kaggle package missing (pip install kaggle)")
    except ImportError:
        issues.append("Kaggle package not installed")
        print("   ❌ Kaggle package missing (pip install kaggle)")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nPlease resolve these issues before continuing.")
        return False
    
    print("\n✅ All prerequisites met!")
    return True

async def demo_data_collection():
    """Demonstrate data collection capabilities."""
    print("\n📦 DEMO: Data Collection")
    print("-" * 40)
    
    try:
        # Initialize data collectors
        print("🔄 Initializing data collectors...")
        
        # Kaggle downloader
        print("   📊 Setting up Kaggle downloader...")
        kaggle_downloader = KaggleDataDownloader()
        
        # Get info about a high-priority dataset
        dataset_name = "carrie1/ecommerce-data"
        print(f"   📋 Getting info for dataset: {dataset_name}")
        info = kaggle_downloader.get_dataset_info(dataset_name)
        
        if info:
            print(f"      Title: {info.get('title', 'N/A')}")
            print(f"      Size: {info.get('size', 0):,} bytes")
            print(f"      Files: {info.get('total_files', 0)}")
        
        # Web scraper
        print("   🌐 Setting up web scraper...")
        web_scraper = WebDataScraper()
        
        # Get sample external data
        print("   📅 Getting holiday calendar...")
        holidays_df = web_scraper.get_holiday_calendar()
        if not holidays_df.empty:
            print(f"      Found {len(holidays_df)} holiday entries")
        else:
            print("      No holiday data available (check API keys)")
        
        # WMS extractor (will likely fail in demo environment)
        print("   📦 Testing WMS connection...")
        wms_extractor = WMSDataExtractor()
        print("      WMS extractor initialized (connection not tested)")
        await wms_extractor.close()
        
        print("\n✅ Data collection demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in data collection demo: {e}")
        print("This is normal in a demo environment without full database setup.")

def demo_configuration():
    """Show configuration options."""
    print("\n⚙️  DEMO: Configuration")
    print("-" * 40)
    
    from config import PROPHET_CONFIG, KAGGLE_DATASETS
    
    print("📊 Prophet Model Configuration:")
    base_config = PROPHET_CONFIG.get("base_model", {})
    print(f"   • Growth: {base_config.get('growth', 'linear')}")
    print(f"   • Seasonality Mode: {base_config.get('seasonality_mode', 'multiplicative')}")
    print(f"   • Yearly Seasonality: {base_config.get('yearly_seasonality', True)}")
    print(f"   • Weekly Seasonality: {base_config.get('weekly_seasonality', True)}")
    
    print("\n📦 Available Kaggle Datasets:")
    high_priority = KAGGLE_DATASETS.get("high_priority", [])
    for i, dataset in enumerate(high_priority[:3], 1):  # Show first 3
        print(f"   {i}. {dataset['name']}")
        print(f"      Description: {dataset['description']}")
    
    if len(high_priority) > 3:
        print(f"   ... and {len(high_priority) - 3} more datasets")

async def demo_orchestrator():
    """Demonstrate the data orchestrator (limited demo)."""
    print("\n🎼 DEMO: Data Orchestration")
    print("-" * 40)
    
    try:
        SeasonalDataOrchestrator()
        print("✅ Data orchestrator initialized")
        
        print("📋 Available collection methods:")
        print("   • collect_kaggle_datasets() - Download and process Kaggle data")
        print("   • collect_external_data() - Scrape external market data")
        print("   • collect_wms_data() - Extract WMS historical data")
        print("   • orchestrate_full_data_collection() - Run complete pipeline")
        
        print("\n💡 To run full data collection:")
        print("   results = await orchestrator.orchestrate_full_data_collection()")
        
    except Exception as e:
        print(f"❌ Error initializing orchestrator: {e}")

def show_next_steps():
    """Show next steps for users."""
    print("\n🎯 NEXT STEPS")
    print("-" * 40)
    
    print("1. 📋 Configure API Keys:")
    print("   • Set up Kaggle API credentials")
    print("   • Configure weather API key (optional)")
    print("   • Set up economic data API keys (optional)")
    
    print("\n2. 📊 Run Data Collection:")
    print("   python -c \"import asyncio; from src.data_orchestrator import SeasonalDataOrchestrator; asyncio.run(SeasonalDataOrchestrator().orchestrate_full_data_collection())\"")
    
    print("\n3. 🔧 Train Prophet Models:")
    print("   • Implement model training pipeline (coming soon)")
    print("   • Configure seasonality parameters")
    print("   • Set up cross-validation")
    
    print("\n4. 📈 Create Visualizations:")
    print("   • Implement forecast charts")
    print("   • Build interactive dashboards")
    print("   • Create business KPI displays")
    
    print("\n5. 🔗 Integrate with WMS:")
    print("   • Connect to WMS database")
    print("   • Create API endpoints")
    print("   • Update frontend dashboards")
    
    print("\n📚 Documentation:")
    print("   • README.md - Module overview")
    print("   • IMPLEMENTATION_GUIDE.md - Step-by-step guide")
    print("   • config.py - Configuration options")
    
    print("\n🎨 Visualizations:")
    print("   python architecture_visualization.py")

async def main():
    """Main function to run the quick start demo."""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please resolve issues and try again.")
        return
    
    # Run demos
    await demo_data_collection()
    demo_configuration()
    await demo_orchestrator()
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 Quick start demo completed!")
    print("You're ready to begin implementing seasonal inventory forecasting!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")
