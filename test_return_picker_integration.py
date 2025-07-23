#!/usr/bin/env python3
"""
Simple test to verify that approved returns appear in the picker dashboard.
This tests the corrected implementation using only the 'receiving' collection.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8002"
AUTH_USERNAME = "manager"
AUTH_PASSWORD = "manager123"

def test_return_workflow():
    """Test the return workflow with simplified collection structure"""
    
    print("🧪 Testing Return → Picker Dashboard Integration")
    print("=" * 55)
    
    # Step 1: Login
    print("1️⃣ Authenticating...")
    auth_response = requests.post(f"{BASE_URL}/api/v1/auth/login", data={
        "username": AUTH_USERNAME,
        "password": AUTH_PASSWORD
    })
    
    if auth_response.status_code != 200:
        print(f"❌ Authentication failed: {auth_response.status_code}")
        return False
    
    token = auth_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Step 2: Check initial picker dashboard state
    print("\n2️⃣ Checking initial picker dashboard...")
    picker_response = requests.get(f"{BASE_URL}/api/v1/receiving/items/by-status", headers=headers)
    
    if picker_response.status_code == 200:
        initial_data = picker_response.json()
        initial_count = len(initial_data.get("items_available_for_storing", []))
        print(f"📊 Initial items available for storing: {initial_count}")
    else:
        print(f"⚠️ Failed to get picker dashboard: {picker_response.status_code}")
        print(picker_response.text)
        initial_count = 0
    
    # Step 3: Create a test return
    print("\n3️⃣ Creating test return...")
    return_data = {
        "orderID": 1,
        "customerID": 1,
        "workerID": 1,
        "return_method": "customer_drop_off",
        "items": [
            {
                "itemID": 1,
                "orderDetailID": 1,
                "quantity": 1,
                "reason": "Testing return workflow",
                "condition": "new"
            }
        ]
    }
    
    create_response = requests.post(f"{BASE_URL}/api/v1/returns/", json=return_data, headers=headers)
    
    if create_response.status_code != 201:
        print(f"❌ Return creation failed: {create_response.status_code}")
        print(create_response.text)
        return False
    
    return_record = create_response.json()
    return_id = return_record["returnID"]
    print(f"✅ Return created with ID: {return_id}")
    
    # Step 4: Approve the return
    print(f"\n4️⃣ Approving return {return_id}...")
    approve_response = requests.put(
        f"{BASE_URL}/api/v1/returns/{return_id}",
        json={"status": "approved"},
        headers=headers
    )
    
    if approve_response.status_code != 200:
        print(f"❌ Return approval failed: {approve_response.status_code}")
        print(approve_response.text)
        return False
    
    approved_return = approve_response.json()
    print(f"✅ Return {return_id} approved")
    
    # Check if inventory increase was created
    if "inventory_increase_created" in approved_return:
        increase_info = approved_return["inventory_increase_created"]
        print(f"📦 Inventory increase created: {increase_info}")
    
    # Step 5: Check picker dashboard after approval
    print(f"\n5️⃣ Checking picker dashboard after approval...")
    picker_response_after = requests.get(f"{BASE_URL}/api/v1/receiving/items/by-status", headers=headers)
    
    if picker_response_after.status_code == 200:
        final_data = picker_response_after.json()
        final_count = len(final_data.get("items_available_for_storing", []))
        print(f"📊 Final items available for storing: {final_count}")
        
        # Look for our return items
        return_items = []
        for item in final_data.get("items_available_for_storing", []):
            if (str(return_id) in str(item.get("return_id", "")) or 
                str(return_id) in str(item.get("original_return_id", "")) or
                "return" in item.get("notes", "").lower()):
                return_items.append(item)
                print(f"   🔄 RETURN ITEM FOUND:")
                print(f"      Name: {item.get('itemName', item.get('item_name', 'Unknown'))}")
                print(f"      Quantity: {item.get('quantity')}")
                print(f"      Notes: {item.get('notes', 'No notes')}")
                print(f"      Type: {item.get('type', 'unknown')}")
        
        if return_items:
            print(f"\n🎉 SUCCESS! Found {len(return_items)} return item(s) in picker dashboard!")
            print(f"Items increased from {initial_count} to {final_count}")
            return True
        else:
            print(f"\n❌ FAILED! No return items found in picker dashboard.")
            print("Items available for storing:")
            for item in final_data.get("items_available_for_storing", []):
                print(f"   - {item.get('itemName', item.get('item_name', 'Unknown'))} (Notes: {item.get('notes', 'No notes')})")
            return False
    else:
        print(f"⚠️ Failed to get picker dashboard after approval: {picker_response_after.status_code}")
        print(picker_response_after.text)
        return False

if __name__ == "__main__":
    success = test_return_workflow()
    if success:
        print("\n✅ Test PASSED - Return items appear in picker dashboard!")
    else:
        print("\n❌ Test FAILED - Return items do not appear in picker dashboard!")
