import requests
import json
from datetime import datetime

# Base URL for the API - Updated based on findings
BASE_URL = "http://localhost:8002/api/v1"

def test_returns_workflow():
    """Test the complete returns workflow"""
    
    print("🧪 Testing Returns Workflow...")
    
    # Step 1: Test health check
    print("\n1. Testing server health check...")
    try:
        response = requests.get("http://localhost:8002/")
        if response.status_code == 200:
            print("✅ Server health check passed")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing health check: {e}")
    
    # Step 2: Login to get auth token
    print("\n2. Getting authentication token...")
    
    # Try different possible authentication endpoints
    auth_endpoints = [
        f"{BASE_URL}/auth/token",  # This one returned 401, so it exists!
        f"{BASE_URL}/token",
        "http://localhost:8002/auth/token",  # Try without /api/v1
        "http://localhost:8002/token"        # Try without /api/v1
    ]
    
    auth_headers = None
    
    for endpoint in auth_endpoints:
        try:
            print(f"   Trying endpoint: {endpoint}")
            
            # Try different username/password combinations
            login_combinations = [
                {"username": "receiver", "password": "receiver123"},  # ReceivingClerk - perfect for returns
                {"username": "manager", "password": "manager123"},    # Manager - can approve returns
                {"username": "picker", "password": "picker123"},     # Picker
                {"username": "packer", "password": "packer123"},     # Packer
                {"username": "driver", "password": "driver123"}      # Driver
            ]
            
            for login_data in login_combinations:
                print(f"     Trying credentials: {login_data['username']}")
                response = requests.post(endpoint, data=login_data)
                
                if response.status_code == 200:
                    token_data = response.json()
                    auth_headers = {"Authorization": f"Bearer {token_data['access_token']}"}
                    print(f"✅ Authentication successful with {login_data['username']}")
                    break
                elif response.status_code == 422:
                    # Try JSON format
                    response = requests.post(endpoint, json=login_data)
                    if response.status_code == 200:
                        token_data = response.json()
                        auth_headers = {"Authorization": f"Bearer {token_data['access_token']}"}
                        print(f"✅ Authentication successful with {login_data['username']} (JSON format)")
                        break
            
            if auth_headers:
                break
            
            print(f"   Status: {response.status_code}")
            
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    if not auth_headers:
        print("❌ All authentication attempts failed")
        print("\n🔍 Let's check what endpoints are available...")
        
        # Check if we can access any endpoint without auth
        try:
            response = requests.get(f"{BASE_URL}/")
            print(f"API base status: {response.status_code}")
        except:
            pass
            
        try:
            response = requests.get(f"{BASE_URL}/docs")
            print(f"API docs status: {response.status_code}")
            if response.status_code == 200:
                print("✅ Try visiting http://localhost:8002/api/v1/docs to see available endpoints")
        except:
            pass
        
        return
    
    # Step 3: Check if we have orders to return
    print("\n3. Checking available orders...")
    try:
        response = requests.get(f"{BASE_URL}/orders/", headers=auth_headers)
        if response.status_code == 200:
            orders = response.json()
            print(f"✅ Found {len(orders)} orders")
            
            # Find a delivered order
            delivered_orders = [order for order in orders if order.get("order_status") == "delivered"]
            if delivered_orders:
                test_order = delivered_orders[0]
                print(f"✅ Found delivered order ID: {test_order['orderID']}")
            else:
                print("⚠️ No delivered orders found, creating test return anyway...")
                test_order = orders[0] if orders else {"orderID": 1, "customerID": 1}
        else:
            print(f"❌ Failed to get orders: {response.status_code}")
            print(f"Response: {response.text}")
            test_order = {"orderID": 1, "customerID": 1}
    except Exception as e:
        print(f"❌ Error getting orders: {e}")
        test_order = {"orderID": 1, "customerID": 1}
    
    # Step 4: Create a return
    print("\n4. Creating a test return...")
    try:
        return_data = {
            "orderID": test_order["orderID"],
            "customerID": test_order.get("customerID", 1),
            "workerID": 2,  # Bob Receiver's workerID (from the workers list we saw)
            "return_date": datetime.now().isoformat(),
            "status": "pending",
            "items": [
                {
                    "itemID": 1,
                    "orderDetailID": 1,  # Assuming first order detail
                    "quantity": 1,
                    "reason": "exchanged",
                    "condition": "good"
                },
                {
                    "itemID": 2,
                    "orderDetailID": 2,  # Assuming second order detail
                    "quantity": 1,
                    "reason": "damaged",
                    "condition": "damaged"
                }
            ]
        }
        
        response = requests.post(f"{BASE_URL}/returns/", 
                               json=return_data, 
                               headers=auth_headers)
        
        if response.status_code == 201:
            created_return = response.json()
            return_id = created_return["returnID"]
            print(f"✅ Return created successfully with ID: {return_id}")
        else:
            print(f"❌ Failed to create return: {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error creating return: {e}")
        return
    
    # Step 5: Get the created return
    print(f"\n5. Retrieving return {return_id}...")
    try:
        response = requests.get(f"{BASE_URL}/returns/{return_id}", headers=auth_headers)
        if response.status_code == 200:
            return_details = response.json()
            print(f"✅ Retrieved return: Status = {return_details.get('status')}")
        else:
            print(f"❌ Failed to retrieve return: {response.status_code}")
    except Exception as e:
        print(f"❌ Error retrieving return: {e}")
    
    # Step 6: Process the return
    print(f"\n6. Processing return {return_id}...")
    try:
        response = requests.post(f"{BASE_URL}/returns/{return_id}/process", 
                               headers=auth_headers)
        
        if response.status_code == 200:
            process_result = response.json()
            print(f"✅ Return processed successfully")
            print(f"   Status: {process_result.get('status')}")
            print(f"   Message: {process_result.get('message')}")
        else:
            print(f"❌ Failed to process return: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error processing return: {e}")
    
    # Step 7: Get all returns to verify
    print("\n7. Getting all returns...")
    try:
        response = requests.get(f"{BASE_URL}/returns/", headers=auth_headers)
        if response.status_code == 200:
            all_returns = response.json()
            print(f"✅ Retrieved {len(all_returns)} total returns")
            
            # Find our test return
            our_return = next((r for r in all_returns if r.get("returnID") == return_id), None)
            if our_return:
                print(f"✅ Our test return found with status: {our_return.get('status')}")
            else:
                print("⚠️ Our test return not found in the list")
        else:
            print(f"❌ Failed to get all returns: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting all returns: {e}")
    
    print("\n🎉 Returns workflow test completed!")

def check_available_endpoints():
    """Helper function to check what endpoints are available"""
    print("🔍 Checking available endpoints...")
    
    base_urls = [
        "http://localhost:8002",
        "http://localhost:8002/api",
        "http://localhost:8002/api/v1"
    ]
    
    for base_url in base_urls:
        try:
            response = requests.get(f"{base_url}/docs")
            if response.status_code == 200:
                print(f"✅ API documentation available at: {base_url}/docs")
        except:
            pass
    
    print("\n💡 Visit the /docs endpoint in your browser to see all available API endpoints")

if __name__ == "__main__":
    test_returns_workflow()
    print("\n" + "="*50)
    check_available_endpoints()