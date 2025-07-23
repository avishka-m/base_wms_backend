#!/usr/bin/env python3
"""
Test script to verify that approved returns appear in the picker dashboard
as "Items Available for Storing".
"""

import asyncio
import requests
import json

# Configuration
BASE_URL = "http://localhost:8002"
AUTH_USERNAME = "manager"
AUTH_PASSWORD = "manager123"

async def test_return_to_storing_workflow():
    """Test the complete workflow from return creation to storing availability"""
    
    print("🧪 Testing Return → Storing Workflow")
    print("=" * 50)
    
    # Step 1: Login to get token
    print("1️⃣ Authenticating...")
    auth_response = requests.post(f"{BASE_URL}/api/v1/auth/login", data={
        "username": AUTH_USERNAME,
        "password": AUTH_PASSWORD
    })
    
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.status_code}")
        print(auth_response.text)
        return
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Step 2: Create a test return
    print("\n2️⃣ Creating test return...")
    return_data = {
        "orderID": 1,
        "customerID": 1,
        "workerID": 1,
        "return_method": "customer_drop_off",
        "items": [
            {
                "itemID": 1,
                "orderDetailID": 1,
                "quantity": 2,
                "reason": "Wrong size",
                "condition": "new"
            },
            {
                "itemID": 2,
                "orderDetailID": 2,
                "quantity": 1,
                "reason": "Damaged",
                "condition": "damaged"
            }
        ]
    }
    
    create_response = requests.post(
        f"{BASE_URL}/api/v1/returns/", 
        json=return_data,
        headers=headers
    )
    
    if create_response.status_code != 201:
        print(f"❌ Return creation failed: {create_response.status_code}")
        print(create_response.text)
        return
    
    return_record = create_response.json()
    return_id = return_record["returnID"]
    print(f"✅ Return created with ID: {return_id}")
    
    # Step 3: Check picker dashboard before approval
    print(f"\n3️⃣ Checking picker dashboard BEFORE approval...")
    picker_response = requests.get(
        f"{BASE_URL}/api/v1/receiving/picker-dashboard",
        headers=headers
    )
    
    if picker_response.status_code == 200:
        dashboard_data = picker_response.json()
        storing_items_before = dashboard_data.get("items_available_for_storing", [])
        print(f"📊 Items available for storing BEFORE: {len(storing_items_before)}")
        for item in storing_items_before:
            print(f"   - {item.get('itemName', 'Unknown')} (ID: {item.get('itemID')}) - {item.get('quantity')} units")
    else:
        print(f"⚠️ Failed to get picker dashboard: {picker_response.status_code}")
    
    # Step 4: Approve the return (this should create storing items)
    print(f"\n4️⃣ Approving return {return_id}...")
    approve_response = requests.put(
        f"{BASE_URL}/api/v1/returns/{return_id}",
        json={"status": "approved"},
        headers=headers
    )
    
    if approve_response.status_code != 200:
        print(f"❌ Return approval failed: {approve_response.status_code}")
        print(approve_response.text)
        return
    
    approved_return = approve_response.json()
    print(f"✅ Return {return_id} approved")
    
    # Check if inventory increase was created
    if "inventory_increase_created" in approved_return:
        increase_info = approved_return["inventory_increase_created"]
        print(f"📦 Inventory increase created: {increase_info['message']}")
    
    # Step 5: Check picker dashboard after approval
    print(f"\n5️⃣ Checking picker dashboard AFTER approval...")
    picker_response_after = requests.get(
        f"{BASE_URL}/api/v1/receiving/picker-dashboard",
        headers=headers
    )
    
    if picker_response_after.status_code == 200:
        dashboard_data_after = picker_response_after.json()
        storing_items_after = dashboard_data_after.get("items_available_for_storing", [])
        print(f"📊 Items available for storing AFTER: {len(storing_items_after)}")
        
        # Look for our return items
        return_items_found = []
        for item in storing_items_after:
            if ("return" in item.get("notes", "").lower() or 
                item.get("type") == "return_processing" or
                str(return_id) in str(item.get("original_return_id", ""))):
                return_items_found.append(item)
                print(f"   🔄 RETURN ITEM: {item.get('itemName', 'Unknown')} (ID: {item.get('itemID')}) - {item.get('quantity')} units")
                print(f"      Notes: {item.get('notes', 'No notes')}")
            else:
                print(f"   - {item.get('itemName', 'Unknown')} (ID: {item.get('itemID')}) - {item.get('quantity')} units")
        
        if return_items_found:
            print(f"\n🎉 SUCCESS! Found {len(return_items_found)} return items in picker dashboard!")
        else:
            print(f"\n❌ FAILED! No return items found in picker dashboard.")
            print("🔍 Let's check the receiving_items collection directly...")
            
            # Debug: Check receiving_items collection directly
            debug_response = requests.get(f"{BASE_URL}/api/v1/debug/receiving-items", headers=headers)
            if debug_response.status_code == 200:
                print("Debug info:", json.dumps(debug_response.json(), indent=2))
    else:
        print(f"⚠️ Failed to get picker dashboard after approval: {picker_response_after.status_code}")
        print(picker_response_after.text)
    
    print(f"\n🏁 Test completed for return {return_id}")

if __name__ == "__main__":
    asyncio.run(test_return_to_storing_workflow())
