#!/usr/bin/env python3
"""
MongoDB Atlas Performance Setup Script

This script sets up all performance optimizations for MongoDB Atlas including:
1. Creating all necessary database indexes
2. Validating Atlas connection settings
3. Testing connection performance
4. Providing performance recommendations

Run this script after configuring your Atlas environment variables.
"""

import os
import time
import sys
from datetime import datetime
from typing import Dict, Any

# Add config path
config_path = os.path.join(os.path.dirname(__file__), 'config')
sys.path.insert(0, config_path)

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Import our optimization modules
from config.atlas_optimization import atlas_optimizer, get_database_url, get_client_options
from config.database_indexes import DatabaseIndexManager, create_all_indexes

def test_connection_performance(mongodb_url: str, database_name: str) -> Dict[str, Any]:
    """
    Test MongoDB connection performance and return metrics.
    
    Args:
        mongodb_url: MongoDB connection string
        database_name: Database name
        
    Returns:
        Performance metrics dictionary
    """
    print("🔍 Testing MongoDB connection performance...")
    
    client_options = get_client_options()
    
    try:
        # Test connection time
        start_time = time.time()
        client = MongoClient(mongodb_url, **client_options)
        
        # Force connection by running a simple command
        db = client[database_name]
        result = db.command('ping')
        connection_time = time.time() - start_time
        
        # Test query performance
        start_time = time.time()
        collections = db.list_collection_names()
        query_time = time.time() - start_time
        
        # Get server info
        server_info = client.server_info()
        
        # Get database stats
        db_stats = db.command('dbStats')
        
        metrics = {
            "connection_time": round(connection_time * 1000, 2),  # Convert to milliseconds
            "query_time": round(query_time * 1000, 2),
            "server_version": server_info.get('version', 'Unknown'),
            "database_size": db_stats.get('dataSize', 0),
            "collection_count": len(collections),
            "is_atlas": 'mongodb+srv' in mongodb_url,
            "connection_successful": True
        }
        
        client.close()
        return metrics
        
    except ConnectionFailure as e:
        return {
            "connection_successful": False,
            "error": f"Connection failed: {str(e)}",
            "error_type": "ConnectionFailure"
        }
    except ServerSelectionTimeoutError as e:
        return {
            "connection_successful": False,
            "error": f"Server selection timeout: {str(e)}",
            "error_type": "TimeoutError"
        }
    except Exception as e:
        return {
            "connection_successful": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "UnknownError"
        }

def validate_atlas_environment() -> bool:
    """
    Validate that all required Atlas environment variables are configured.
    
    Returns:
        True if valid, False otherwise
    """
    print("🔧 Validating Atlas environment configuration...")
    
    required_vars = [
        'ATLAS_USERNAME',
        'ATLAS_PASSWORD', 
        'ATLAS_CLUSTER_HOST',
        'DATABASE_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("\n📝 Please set the following environment variables:")
        print("   ATLAS_USERNAME=your_atlas_username")
        print("   ATLAS_PASSWORD=your_atlas_password")
        print("   ATLAS_CLUSTER_HOST=cluster0.xxxxx.mongodb.net")
        print("   DATABASE_NAME=warehouse_management")
        print("\n💡 You can set these in a .env file or export them directly.")
        return False
    
    print("✅ All required environment variables are configured")
    return True

def create_env_template():
    """Create a .env template file with Atlas configuration."""
    env_template = """# MongoDB Atlas Configuration
# Copy this file to .env and fill in your Atlas credentials

# Atlas Authentication
ATLAS_USERNAME=your_atlas_username
ATLAS_PASSWORD=your_atlas_password
ATLAS_CLUSTER_HOST=cluster0.xxxxx.mongodb.net

# Database Configuration
DATABASE_NAME=warehouse_management
ATLAS_APP_NAME=warehouse-management-system

# Optional: Local fallback (for development)
# If Atlas env vars are not set, will use localhost
# MONGODB_URL=mongodb://localhost:27017

# Application Configuration
SECRET_KEY=your-secret-key-here
DEBUG_MODE=True
ENVIRONMENT=development
"""
    
    env_file_path = os.path.join(os.path.dirname(__file__), '.env.atlas.template')
    
    with open(env_file_path, 'w') as f:
        f.write(env_template)
    
    print(f"📄 Created environment template: {env_file_path}")
    print("   Copy this to .env and configure your Atlas credentials")

def setup_atlas_performance():
    """
    Main function to set up MongoDB Atlas performance optimizations.
    """
    print("🚀 MongoDB Atlas Performance Setup")
    print("=" * 50)
    
    # Step 1: Validate environment
    if not validate_atlas_environment():
        print("\n🔧 Creating environment template...")
        create_env_template()
        return False
    
    # Step 2: Get connection settings
    mongodb_url = get_database_url()
    database_name = os.getenv('DATABASE_NAME', 'warehouse_management')
    
    print(f"\n🔗 Connection URL: {mongodb_url[:50]}{'...' if len(mongodb_url) > 50 else ''}")
    print(f"📊 Database: {database_name}")
    
    # Step 3: Test connection performance
    metrics = test_connection_performance(mongodb_url, database_name)
    
    if not metrics.get('connection_successful', False):
        print(f"❌ Connection test failed: {metrics.get('error', 'Unknown error')}")
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Check your Atlas credentials")
        print("   2. Verify cluster hostname")
        print("   3. Ensure your IP is whitelisted in Atlas")
        print("   4. Check network connectivity")
        return False
    
    # Step 4: Display performance metrics
    print("\n📊 Connection Performance Metrics:")
    print(f"   Connection Time: {metrics['connection_time']}ms")
    print(f"   Query Time: {metrics['query_time']}ms")
    print(f"   Server Version: {metrics['server_version']}")
    print(f"   Collection Count: {metrics['collection_count']}")
    print(f"   Database Size: {metrics.get('database_size', 0):,} bytes")
    print(f"   Atlas Connection: {'Yes' if metrics['is_atlas'] else 'No'}")
    
    # Step 5: Performance assessment
    connection_time = metrics['connection_time']
    if connection_time > 5000:  # 5 seconds
        print(f"⚠️  Slow connection time ({connection_time}ms). Consider:")
        print("   - Choosing a closer Atlas region")
        print("   - Checking network latency")
    elif connection_time > 1000:  # 1 second
        print(f"⚠️  Moderate connection time ({connection_time}ms)")
    else:
        print(f"✅ Good connection time ({connection_time}ms)")
    
    # Step 6: Create database indexes
    print(f"\n🗂️  Creating database indexes for performance optimization...")
    try:
        results = create_all_indexes(mongodb_url, database_name)
        
        total_indexes = sum(len(indexes) for indexes in results.values())
        print(f"✅ Successfully created {total_indexes} indexes across {len(results)} collections")
        
        # Show index summary
        for collection, indexes in results.items():
            if indexes:  # Only show collections where indexes were created
                print(f"   📁 {collection}: {len(indexes)} indexes")
    
    except Exception as e:
        print(f"❌ Failed to create indexes: {e}")
        return False
    
    # Step 7: Final recommendations
    print(f"\n🎯 Performance Optimization Complete!")
    print(f"=" * 50)
    print(f"✅ Atlas connection optimized")
    print(f"✅ Database indexes created")
    print(f"✅ Connection pooling configured")
    print(f"✅ Compression enabled")
    
    print(f"\n💡 Additional Recommendations:")
    if metrics['is_atlas']:
        print(f"   🌍 Ensure Atlas cluster is in the same region as your application")
        print(f"   📈 Monitor performance in Atlas dashboard")
        print(f"   🔍 Use Atlas Performance Advisor for query optimization")
    
    print(f"   🚀 Connection time: {connection_time}ms (target: <1000ms)")
    print(f"   📊 Monitor slow queries (>100ms) in your logs")
    print(f"   🔄 Consider read replicas for read-heavy workloads")
    
    return True

if __name__ == "__main__":
    print("MongoDB Atlas Performance Setup")
    print("This script optimizes your MongoDB Atlas connection for the Warehouse Management System")
    print()
    
    success = setup_atlas_performance()
    
    if success:
        print(f"\n🎉 Setup completed successfully!")
        print(f"Your application should now have significantly improved Atlas performance.")
    else:
        print(f"\n❌ Setup failed. Please review the errors above and try again.")
        sys.exit(1) 