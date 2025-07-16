#!/usr/bin/env python3
"""
Test WebSocket connection
"""
import asyncio
import websockets
import json
import requests

# Base URL for the API
BASE_URL = "http://localhost:8002"
WS_URL = "ws://localhost:8002"

async def test_websocket():
    """Test WebSocket connection"""
    try:
        # First get an auth token
        login_data = {
            "username": "manager",
            "password": "manager123"
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/auth/token", data=login_data)
        if response.status_code != 200:
            print(f"❌ Login failed: {response.status_code} - {response.text}")
            return
        
        token = response.json().get("access_token")
        print("✅ Login successful, token obtained")
        
        # Try to connect to WebSocket
        ws_url = f"{WS_URL}/api/v1/ws/orders?token={token}"
        print(f"🔗 Attempting WebSocket connection to: {ws_url}")
        
        try:
            async with websockets.connect(ws_url) as websocket:
                print("✅ WebSocket connection established!")
                
                # Send a test message or wait for initial response
                try:
                    # Wait for welcome message or initial data
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📨 Received: {response}")
                except asyncio.TimeoutError:
                    print("⏱️ No initial message received (this is normal)")
                
                print("✅ WebSocket test completed successfully")
                
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"❌ WebSocket connection closed: {e}")
        except Exception as e:
            print(f"❌ WebSocket connection error: {e}")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    print("Testing WebSocket connection...")
    asyncio.run(test_websocket())
