"""
MongoDB Atlas Optimization Configuration

This module provides optimized connection settings and configurations
specifically designed for MongoDB Atlas cloud deployments to resolve
performance issues and improve connection reliability.
"""

import os
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AtlasConnectionOptimizer:
    """
    MongoDB Atlas connection optimizer with performance enhancements.
    Addresses common Atlas performance issues including:
    - Network latency optimization
    - Connection pooling for cloud environments
    - Compression to reduce bandwidth usage
    - Proper timeout settings for cloud connections
    - Read preference optimization for replica sets
    """
    
    def __init__(self):
        self.username = os.getenv("ATLAS_USERNAME")
        self.password = os.getenv("ATLAS_PASSWORD") 
        self.cluster_host = os.getenv("ATLAS_CLUSTER_HOST")
        self.database_name = os.getenv("DATABASE_NAME", "warehouse_management")
        self.app_name = os.getenv("ATLAS_APP_NAME", "warehouse-management-system")
        
    def _get_available_compressors(self):
        """Detect available compression libraries and return them in order of preference."""
        compressors = []
        
        # Check for snappy (best compression)
        try:
            import snappy
            compressors.append('snappy')
        except ImportError:
            pass
        
        # zlib is built into Python
        compressors.append('zlib')
        
        return compressors
        
    def get_optimized_connection_string(self) -> str:
        """
        Generate optimized MongoDB Atlas connection string with performance settings.
        
        Returns:
            Optimized connection string for Atlas
        """
        if not all([self.username, self.password, self.cluster_host]):
            raise ValueError(
                "Missing required Atlas environment variables: "
                "ATLAS_USERNAME, ATLAS_PASSWORD, ATLAS_CLUSTER_HOST"
            )
        
        # URL encode credentials to handle special characters
        username = quote_plus(self.username)
        password = quote_plus(self.password)
        
        # Get available compressors
        compressors = self._get_available_compressors()
        compressor_string = ",".join(compressors)
        
        # Optimized connection string for Atlas
        connection_string = (
            f"mongodb+srv://{username}:{password}@{self.cluster_host}/"
            f"{self.database_name}?"
            f"retryWrites=true&"
            f"w=majority&"
            f"appName={self.app_name}&"
            f"compressors={compressor_string}&"  # Dynamic compression
            f"readPreference=secondaryPreferred&"  # Use secondaries for reads
            f"maxStalenessSeconds=90&"  # Allow slight staleness for performance
            f"readConcernLevel=local&"  # Faster read concern
            f"connectTimeoutMS=30000&"  # 30 seconds for initial connection
            f"socketTimeoutMS=30000&"  # 30 seconds for socket operations
            f"serverSelectionTimeoutMS=30000&"  # 30 seconds for server selection
            f"heartbeatFrequencyMS=10000&"  # 10 seconds heartbeat
            f"maxIdleTimeMS=60000&"  # 60 seconds max idle time
            f"waitQueueTimeoutMS=10000"  # 10 seconds wait queue timeout
        )
        
        return connection_string
    
    def get_optimized_client_options(self) -> Dict[str, Any]:
        """
        Get optimized client options for Atlas connections.
        
        Returns:
            Dictionary of client options optimized for Atlas
        """
        return {
            # Connection Pool Settings (optimized for Atlas)
            'maxPoolSize': 100,  # Increased for Atlas
            'minPoolSize': 10,   # Higher minimum for Atlas
            'maxIdleTimeMS': 60000,  # 60 seconds
            'waitQueueTimeoutMS': 10000,  # 10 seconds
            'maxConnecting': 5,  # Limit concurrent connections
            
            # Timeout Settings (longer for cloud)
            'connectTimeoutMS': 30000,  # 30 seconds
            'socketTimeoutMS': 30000,   # 30 seconds
            'serverSelectionTimeoutMS': 30000,  # 30 seconds
            'heartbeatFrequencyMS': 10000,  # 10 seconds
            
            # Retry and Reliability
            'retryWrites': True,
            'retryReads': True,
            
            # Compression (reduces bandwidth usage) - auto-detect available compressors
            'compressors': self._get_available_compressors(),
            
            # Read Preferences (utilize replica set)
            'readPreference': 'secondaryPreferred',
            'maxStalenessSeconds': 90,
            'readConcernLevel': 'local',
            
            # Write Concerns
            'w': 'majority',
            'wtimeoutMS': 5000,
            
            # SSL/TLS (required for Atlas)
            'tls': True,
            'tlsAllowInvalidCertificates': False,
            
            # Application Identification
            'appName': self.app_name,
        }
    
    def get_fallback_connection_string(self) -> str:
        """
        Get fallback localhost connection string for development.
        
        Returns:
            Localhost MongoDB connection string
        """
        return f"mongodb://localhost:27017/{self.database_name}"
    
    def validate_atlas_connection(self) -> bool:
        """
        Validate that all required Atlas environment variables are set.
        
        Returns:
            True if all variables are set, False otherwise
        """
        required_vars = ['ATLAS_USERNAME', 'ATLAS_PASSWORD', 'ATLAS_CLUSTER_HOST']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠️  Missing Atlas environment variables: {missing_vars}")
            print("   Using localhost fallback connection")
            return False
        
        return True

# Global optimizer instance
atlas_optimizer = AtlasConnectionOptimizer()

# Connection string selection based on environment
def get_database_url() -> str:
    """
    Get the appropriate database URL based on environment configuration.
    
    Returns:
        Optimized MongoDB connection string
    """
    # Check if Atlas environment is configured
    if atlas_optimizer.validate_atlas_connection():
        return atlas_optimizer.get_optimized_connection_string()
    else:
        # Fallback to localhost for development
        return atlas_optimizer.get_fallback_connection_string()

def get_client_options() -> Dict[str, Any]:
    """
    Get optimized client options based on connection type.
    
    Returns:
        Dictionary of MongoDB client options
    """
    if atlas_optimizer.validate_atlas_connection():
        return atlas_optimizer.get_optimized_client_options()
    else:
        # Simplified options for localhost
        return {
            'maxPoolSize': 20,
            'minPoolSize': 2,
            'maxIdleTimeMS': 30000,
            'connectTimeoutMS': 10000,
            'serverSelectionTimeoutMS': 5000,
        }

# Atlas Performance Tips
ATLAS_PERFORMANCE_TIPS = """
MongoDB Atlas Performance Optimization Tips:

1. Geographic Proximity:
   - Choose Atlas cluster region closest to your application
   - Consider multiple regions for global applications

2. Network Optimization:
   - Use compression (enabled in this config)
   - Enable connection pooling (configured here)
   - Use read preferences to distribute load

3. Indexing Strategy:
   - Create indexes on frequently queried fields
   - Use compound indexes for multi-field queries
   - Monitor slow queries in Atlas dashboard

4. Query Optimization:
   - Use projection to limit returned fields
   - Implement pagination for large result sets
   - Use aggregation pipelines efficiently

5. Connection Management:
   - Reuse connections (connection pooling enabled)
   - Monitor connection metrics in Atlas
   - Use appropriate timeout settings
""" 