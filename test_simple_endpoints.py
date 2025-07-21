import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8002"

def test_endpoints():
    """Test available endpoints"""
    
    print("🧪 Testing Available Endpoints...")
    
    # Test main endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test returns endpoint without auth
    print("\n2. Testing returns endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/returns/")
        print(f"Returns endpoint: {response.status_code}")
        if response.status_code == 401:
            print("✅ Authentication required (expected)")
        elif response.status_code == 200:
            print("✅ Returns endpoint accessible")
            print(f"Response: {response.json()}")
        else:
            print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Check docs
    print("\n3. Testing docs endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ Docs endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_endpoints()
