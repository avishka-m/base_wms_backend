import aiohttp
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Use absolute import for chatbot config
from app.config import (
    WMS_API_BASE_URL,
    API_ENDPOINTS
)

logger = logging.getLogger("wms_chatbot.api_client")

class AsyncAPIClient:
    """
    Async client for interacting with the WMS API.
    Provides methods for common API operations using aiohttp.
    """
    
    def __init__(self):
        """Initialize the API client."""
        self.base_url = WMS_API_BASE_URL
        self.endpoints = API_ENDPOINTS
        self._session = None
        self._timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        logger.info(f"Async API Client initialized with base URL: {self.base_url}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout,
                headers=self.get_headers()
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        return {
            "Content-Type": "application/json"
        }
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        logger.debug(f"Async GET request URL: {url} with params: {params}")
        
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error (GET): HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"API timeout (GET): {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"API error (GET): {e}")
            return {"error": str(e)}
    
    async def get_by_id(self, endpoint: str, item_id: int) -> Dict[str, Any]:
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
        logger.debug(f"Async GET by ID request URL: {url}")
        
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error (GET by ID): HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"API timeout (GET by ID): {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"API error (GET by ID): {e}")
            return {"error": str(e)}
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
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
            session = await self._get_session()
            async with session.post(url, json=data) as response:
                if response.status in [200, 201]:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error (POST): HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"API timeout (POST): {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"API error (POST): {e}")
            return {"error": str(e)}
    
    async def put(self, endpoint: str, item_id: int, data: Dict[str, Any]) -> Dict[str, Any]:
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
            session = await self._get_session()
            async with session.put(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error (PUT): HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"API timeout (PUT): {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"API error (PUT): {e}")
            return {"error": str(e)}
    
    async def delete(self, endpoint: str, item_id: int) -> Dict[str, Any]:
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
            session = await self._get_session()
            async with session.delete(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error (DELETE): HTTP {response.status} - {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except asyncio.TimeoutError:
            logger.error(f"API timeout (DELETE): {url}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"API error (DELETE): {e}")
            return {"error": str(e)}
    
    # Specialized methods for common operations
    
    async def get_inventory(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get inventory items with optional filtering."""
        result = await self.get("inventory", params)
        if isinstance(result, dict) and "error" in result:
            return result
        return result if isinstance(result, list) else []
    
    async def get_inventory_item(self, item_id: int) -> Dict[str, Any]:
        """Get a specific inventory item by ID."""
        return await self.get_by_id("inventory", item_id)
    
    async def get_orders(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get orders with optional filtering."""
        result = await self.get("orders", params)
        if isinstance(result, dict) and "error" in result:
            return result
        return result if isinstance(result, list) else []
    
    async def get_order(self, order_id: int) -> Dict[str, Any]:
        """Get a specific order by ID."""
        return await self.get_by_id("orders", order_id)
    
    async def get_workers(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get workers with optional filtering."""
        result = await self.get("workers", params)
        if isinstance(result, dict) and "error" in result:
            return result
        return result if isinstance(result, list) else []
    
    async def get_locations(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get locations with optional filtering."""
        result = await self.get("locations", params)
        if isinstance(result, dict) and "error" in result:
            return result
        return result if isinstance(result, list) else []

# Create global async instance
async_api_client = AsyncAPIClient()

# Legacy sync wrapper for backward compatibility (deprecated)
class APIClient:
    """
    DEPRECATED: Synchronous wrapper around AsyncAPIClient.
    Use async_api_client directly for better performance.
    """
    
    def __init__(self):
        self.async_client = async_api_client
        logger.warning("Using deprecated synchronous APIClient. Switch to async_api_client for better performance.")
    
    def _run_async(self, coro):
        """Run async function in sync context (not recommended)."""
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't use asyncio.run()
            # This is a problematic situation that should be avoided
            logger.error("Cannot run sync API call from within async context. Use async_api_client instead.")
            return {"error": "Cannot run synchronous API call from async context"}
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(coro)
    
    def get_inventory(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._run_async(self.async_client.get_inventory(params))
    
    def get_inventory_item(self, item_id: int) -> Dict[str, Any]:
        return self._run_async(self.async_client.get_inventory_item(item_id))
    
    def get_orders(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._run_async(self.async_client.get_orders(params))
    
    def get_order(self, order_id: int) -> Dict[str, Any]:
        return self._run_async(self.async_client.get_order(order_id))

# Keep backward compatibility
api_client = APIClient()