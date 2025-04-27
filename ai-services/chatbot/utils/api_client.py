import requests
from typing import Dict, Any, List, Optional
import json

# Use absolute import instead of relative import
from config import (
    WMS_API_BASE_URL,
    WMS_API_USERNAME,
    WMS_API_PASSWORD,
    API_ENDPOINTS
)

class APIClient:
    """
    Client for interacting with the WMS API.
    Handles authentication and provides methods for common API operations.
    """
    
    def __init__(self):
        """Initialize the API client."""
        self.base_url = WMS_API_BASE_URL
        self.endpoints = API_ENDPOINTS
        self.token = None
    
    def authenticate(self) -> bool:
        """
        Authenticate with the WMS API and get a JWT token.
        
        Returns:
            Boolean indicating if authentication was successful
        """
        auth_url = self.endpoints["auth"]
        
        try:
            response = requests.post(
                auth_url,
                data={
                    "username": WMS_API_USERNAME,
                    "password": WMS_API_PASSWORD
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            response.raise_for_status()
            data = response.json()
            self.token = data.get("access_token")
            
            return bool(self.token)
        
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get headers with authentication token.
        Authenticates if no token is available.
        
        Returns:
            Dictionary of headers
        """
        if not self.token:
            self.authenticate()
        
        return {
            "Authorization": f"Bearer {self.token}",
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
        
        url = self.endpoints[endpoint]
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                self.authenticate()
                return self.get(endpoint, params)
            raise
    
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
        
        url = f"{self.endpoints[endpoint]}/{item_id}"
        
        try:
            response = requests.get(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                self.authenticate()
                return self.get_by_id(endpoint, item_id)
            raise
    
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
        
        url = self.endpoints[endpoint]
        
        try:
            response = requests.post(
                url,
                json=data,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                self.authenticate()
                return self.post(endpoint, data)
            raise
    
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
        
        url = f"{self.endpoints[endpoint]}/{item_id}"
        
        try:
            response = requests.put(
                url,
                json=data,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                self.authenticate()
                return self.put(endpoint, item_id, data)
            raise
    
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
        
        url = f"{self.endpoints[endpoint]}/{item_id}"
        
        try:
            response = requests.delete(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to re-authenticate
                self.token = None
                self.authenticate()
                return self.delete(endpoint, item_id)
            raise
    
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