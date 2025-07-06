import requests
import logging
from typing import Dict, Any, List, Optional

# Use absolute import for chatbot config
from app.config import (
    WMS_API_BASE_URL,
    API_ENDPOINTS
)

logger = logging.getLogger("wms_chatbot.api_client")

class APIClient:
    """
    Client for interacting with the WMS API.
    Provides methods for common API operations.
    """
    
    def __init__(self):
        """Initialize the API client."""
        self.base_url = WMS_API_BASE_URL
        self.endpoints = API_ENDPOINTS
        logger.info(f"API Client initialized with base URL: {self.base_url}")
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        return {
            "Content-Type": "application/json"
        }
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the WMS API.
        
        Args:
            endpoint: API endpoint name (e.g., "inventory", "orders")
            params: Optional query parameters
            
        Returns:
            API response as dictionary
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = self.base_url + self.endpoints[endpoint]
        logger.debug(f"GET request URL: {url} with params: {params}")
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error (GET): {e}")
            return {"error": str(e)}
    
    def get_by_id(self, endpoint: str, item_id: int) -> Dict[str, Any]:
        """
        Get an item by ID from the WMS API.
        
        Args:
            endpoint: API endpoint name
            item_id: ID of the item to retrieve
            
        Returns:
            API response as dictionary
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = f"{self.base_url}{self.endpoints[endpoint]}/{item_id}"
        logger.debug(f"GET by ID request URL: {url}")
        
        try:
            response = requests.get(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error (GET by ID): {e}")
            return {"error": str(e)}
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the WMS API.
        
        Args:
            endpoint: API endpoint name
            data: Data to send in the request body
            
        Returns:
            API response as dictionary
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = self.base_url + self.endpoints[endpoint]
        
        try:
            response = requests.post(
                url,
                json=data,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error (POST): {e}")
            return {"error": str(e)}
    
    def put(self, endpoint: str, item_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request to the WMS API.
        
        Args:
            endpoint: API endpoint name
            item_id: ID of the item to update
            data: Data to send in the request body
            
        Returns:
            API response as dictionary
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = f"{self.base_url}{self.endpoints[endpoint]}/{item_id}"
        
        try:
            response = requests.put(
                url,
                json=data,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error (PUT): {e}")
            return {"error": str(e)}
    
    def delete(self, endpoint: str, item_id: int) -> Dict[str, Any]:
        """
        Make a DELETE request to the WMS API.
        
        Args:
            endpoint: API endpoint name
            item_id: ID of the item to delete
            
        Returns:
            API response as dictionary
        """
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = f"{self.base_url}{self.endpoints[endpoint]}/{item_id}"
        
        try:
            response = requests.delete(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error (DELETE): {e}")
            return {"error": str(e)}
    
    # Specialized methods for common operations
    
    def get_inventory(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get inventory items with optional filtering."""
        return self.get("inventory", params)
    
    def get_inventory_item(self, item_id: int) -> Dict[str, Any]:
        """Get a specific inventory item by ID."""
        return self.get_by_id("inventory", item_id)
    
    def get_orders(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get orders with optional filtering."""
        return self.get("orders", params)
    
    def get_order(self, order_id: int) -> Dict[str, Any]:
        """Get a specific order by ID."""
        return self.get_by_id("orders", order_id)
    
    def get_workers(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get workers with optional filtering."""
        return self.get("workers", params)
    
    def get_locations(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get storage locations with optional filtering."""
        return self.get("locations", params)

# Initialize singleton instance
api_client = APIClient()