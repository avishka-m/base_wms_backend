import requests

# Test receiver access to inventory
BASE_URL = "http://localhost:8002/api/v1"

def test_receiver_inventory_access():
    print("🧪 Testing receiver access to inventory...")
    
    # Step 1: Login as receiver
    login_data = {"username": "receiver", "password": "receiver123"}
    response = requests.post(f"{BASE_URL}/auth/token", data=login_data)
    
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        return
    
    token_data = response.json()
    auth_headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    print("✅ Login successful")
    
    # Step 2: Test inventory access
    response = requests.get(f"{BASE_URL}/inventory/?limit=1000", headers=auth_headers)
    
    if response.status_code == 200:
        inventory = response.json()
        print(f"✅ Inventory access successful - found {len(inventory)} items")
    else:
        print(f"❌ Inventory access failed: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    test_receiver_inventory_access()
