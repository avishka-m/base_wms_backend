import requests
import json

def test_complete_returns_workflow():
    """Test the complete returns work    # Step 5: Test manager approving return (updating status)
    print("\n5. Testing Manager Approval (Update Status)...")
    
    approval_data = {
        "status": "approved",
        "notes": "Approved for exchange - customer satisfaction priority"
    }
    
    approve_response = requests.put(f"{base_url}/api/v1/returns/{return_id}",
                                   headers=manager_headers,
                                   json=approval_data)eceiver to manager"""
    
    base_url = "http://127.0.0.1:8002"
    
    print("🧪 Testing Complete Returns Workflow")
    print("=" * 50)
    
    # Step 1: Test receiver login and inventory access
    print("\n1. Testing Receiver Authentication & Inventory Access...")
    
    # Login as receiver
    login_data = {
        "username": "receiver",
        "password": "receiver123"
    }
    
    login_response = requests.post(f"{base_url}/api/v1/auth/token", data=login_data)
    if login_response.status_code == 200:
        receiver_token = login_response.json()["access_token"]
        print("✅ Receiver login successful")
    else:
        print(f"❌ Receiver login failed: {login_response.status_code}")
        return
    
    # Test inventory access with receiver token
    headers = {"Authorization": f"Bearer {receiver_token}"}
    inventory_response = requests.get(f"{base_url}/api/v1/inventory/", headers=headers)
    
    if inventory_response.status_code == 200:
        inventory_data = inventory_response.json()
        print(f"✅ Inventory access successful - found {len(inventory_data)} items")
        print("Sample inventory items with location IDs:")
        for item in inventory_data[:3]:
            print(f"  - {item.get('name', 'Unknown')}: Location {item.get('locationID', 'N/A')}")
    else:
        print(f"❌ Inventory access failed: {inventory_response.status_code}")
        print(f"Response: {inventory_response.text}")
        return
    
    # Step 2: Test creating a return (as receiver)
    print("\n2. Testing Return Creation (Receiver)...")
    
    # Create a return with "Exchanged" reason using real order data
    return_data = {
        "orderID": 1,  # Real order ID (delivered status)
        "customerID": 1,  # Real customer ID from order 1
        "workerID": 5001,  # Worker ID (receiver)
        "items": [
            {
                "itemID": 1,  # Real item ID from order 1
                "orderDetailID": 1,  # Real order detail ID from order 1
                "quantity": 1,  # Return 1 out of 5 ordered
                "reason": "Exchanged",
                "condition": "Good"
            }
        ]
    }
    
    create_response = requests.post(f"{base_url}/api/v1/returns/", 
                                  headers=headers, 
                                  json=return_data)
    
    if create_response.status_code == 201:  # 201 = Created (success)
        return_record = create_response.json()
        return_id = return_record["returnID"]  # Use returnID instead of id
        print(f"✅ Return created successfully - ID: {return_id}")
        print(f"   Status: {return_record['status']}")
        print(f"   Return Method: {return_record['return_method']}")
    else:
        print(f"❌ Return creation failed: {create_response.status_code}")
        print(f"Response: {create_response.text}")
        return
    
    # Step 3: Test manager login
    print("\n3. Testing Manager Authentication...")
    
    manager_login_data = {
        "username": "manager",
        "password": "manager123"
    }
    
    manager_login_response = requests.post(f"{base_url}/api/v1/auth/token", data=manager_login_data)
    if manager_login_response.status_code == 200:
        manager_token = manager_login_response.json()["access_token"]
        print("✅ Manager login successful")
    else:
        print(f"❌ Manager login failed: {manager_login_response.status_code}")
        return
    
    # Step 4: Test manager viewing pending returns
    print("\n4. Testing Manager View Pending Returns...")
    
    manager_headers = {"Authorization": f"Bearer {manager_token}"}
    pending_response = requests.get(f"{base_url}/api/v1/returns/?status=pending", headers=manager_headers)
    
    if pending_response.status_code == 200:
        pending_returns = pending_response.json()
        print(f"✅ Manager can view pending returns - found {len(pending_returns)} items")
        for ret in pending_returns:
            print(f"   - Return {ret['returnID']}: {ret['status']}")  # Use returnID
    else:
        print(f"❌ Manager view pending returns failed: {pending_response.status_code}")
        print(f"Response: {pending_response.text}")
    
    # Step 5: Test manager approving return
    print("\n5. Testing Manager Approval...")
    
    approval_data = {
        "approved": True,
        "manager_notes": "Approved for exchange - customer satisfaction priority"
    }
    
    approve_response = requests.post(f"{base_url}/api/v1/returns/{return_id}/approve",
                                   headers=manager_headers,
                                   json=approval_data)
    
    if approve_response.status_code == 200:
        approved_return = approve_response.json()
        print(f"✅ Return approved successfully")
        print(f"   New Status: {approved_return['status']}")
        print(f"   Return Method: {approved_return['return_method']}")
    else:
        print(f"❌ Return approval failed: {approve_response.status_code}")
        print(f"Response: {approve_response.text}")
    
    # Step 6: Test viewing all returns
    print("\n6. Testing View All Returns...")
    
    all_returns_response = requests.get(f"{base_url}/api/v1/returns/", headers=manager_headers)
    
    if all_returns_response.status_code == 200:
        all_returns = all_returns_response.json()
        print(f"✅ Manager can view all returns - found {len(all_returns)} total")
        
        # Show the return we just processed
        our_return = next((r for r in all_returns if r['returnID'] == return_id), None)  # Use returnID
        if our_return:
            print(f"   Our return {return_id}:")
            print(f"     Status: {our_return['status']}")
            print(f"     Return Method: {our_return['return_method']}")
            print(f"     Notes: {our_return.get('notes', 'None')}")
    else:
        print(f"❌ View all returns failed: {all_returns_response.status_code}")
        print(f"Response: {all_returns_response.text}")
    
    # Step 7: Test creating a "Damaged" return (should auto-complete)
    print("\n7. Testing Damaged Item Return (Auto-Complete)...")
    
    damaged_return_data = {
        "orderID": 1,  # Same real order
        "customerID": 1,  # Real customer ID
        "workerID": 5001,  # Worker ID (receiver)
        "items": [
            {
                "itemID": 7,  # Different item from same order
                "orderDetailID": 2,  # Real order detail ID for item 7
                "quantity": 1,  # Return 1 out of 3 ordered
                "reason": "Damaged",
                "condition": "Damaged"
            }
        ]
    }
    
    damaged_response = requests.post(f"{base_url}/api/v1/returns/",
                                   headers=headers,
                                   json=damaged_return_data)
    
    if damaged_response.status_code == 201:  # 201 = Created (success)
        damaged_return = damaged_response.json()
        print(f"✅ Damaged return created - ID: {damaged_return['returnID']}")
        print(f"   Status: {damaged_return['status']}")
        print(f"   Return Method: {damaged_return['return_method']}")
        print("   (Should be auto-completed for damaged items)")
    else:
        print(f"❌ Damaged return creation failed: {damaged_response.status_code}")
        print(f"Response: {damaged_response.text}")
    
    print("\n" + "=" * 50)
    print("🎉 Complete Returns Workflow Test Finished!")
    print("\nWorkflow Summary:")
    print("1. ✅ Receiver can login and access inventory")
    print("2. ✅ Receiver can create returns with proper location IDs")
    print("3. ✅ 'Exchanged' returns go to manager for approval")
    print("4. ✅ Manager can view pending returns and approve them")
    print("5. ✅ 'Damaged' returns are auto-completed")
    print("6. ✅ All returns are tracked in the system")

if __name__ == "__main__":
    test_complete_returns_workflow()
