#!/usr/bin/env python3
"""
Test script to verify that damaged return items do NOT appear in the picker dashboard,
while good condition items DO appear.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8002"
AUTH_USERNAME = "manager"
AUTH_PASSWORD = "manager123"

def test_damaged_items_filtering():
    """Test that damaged items are filtered out from picker dashboard"""
    
    print("🧪 Testing Damaged Items Filtering")
    print("=" * 45)
    
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
        initial_count = 0
    
    # Step 3: Create a return with MIXED condition items (1 good, 1 damaged)
    print("\n3️⃣ Creating return with mixed condition items...")
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
                "reason": "Wrong size",
                "condition": "new"  # GOOD CONDITION - should appear in picker dashboard
            },
            {
                "itemID": 2,
                "orderDetailID": 2,
                "quantity": 1,
                "reason": "Broken in shipping",
                "condition": "damaged"  # DAMAGED - should NOT appear in picker dashboard
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
    print(f"   📦 Item 1 (ID: 1) - Condition: new (should appear in picker)")
    print(f"   💔 Item 2 (ID: 2) - Condition: damaged (should NOT appear in picker)")
    
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
    
    # Check inventory increase details
    if "inventory_increase_created" in approved_return:
        increase_info = approved_return["inventory_increase_created"]
        print(f"📦 Inventory increase info: {increase_info}")
        
        # Check if message indicates filtering
        if "damaged" in increase_info.get("message", "").lower():
            print("✅ System correctly identified damaged items")
        
        items_count = increase_info.get("items_count", 0)
        damaged_count = increase_info.get("damaged_items_count", 0)
        print(f"   🔍 Items sent to picker: {items_count}")
        print(f"   🔍 Damaged items filtered: {damaged_count}")
        
        if items_count == 1 and damaged_count == 1:
            print("✅ Correct filtering: 1 good item sent to picker, 1 damaged item filtered out")
        else:
            print(f"❌ Unexpected filtering result: expected 1 good + 1 damaged, got {items_count} good + {damaged_count} damaged")
    
    # Step 5: Check picker dashboard after approval
    print(f"\n5️⃣ Checking picker dashboard after approval...")
    picker_response_after = requests.get(f"{BASE_URL}/api/v1/receiving/items/by-status", headers=headers)
    
    if picker_response_after.status_code == 200:
        final_data = picker_response_after.json()
        final_count = len(final_data.get("items_available_for_storing", []))
        print(f"📊 Final items available for storing: {final_count}")
        
        # Look for our return items
        return_items_found = []
        damaged_items_found = []
        
        for item in final_data.get("items_available_for_storing", []):
            if (str(return_id) in str(item.get("return_id", "")) or 
                str(return_id) in str(item.get("original_return_id", "")) or
                f"return #{return_id}" in item.get("notes", "").lower()):
                
                item_id = item.get("itemID")
                if item_id == 1:  # Good condition item
                    return_items_found.append(item)
                    print(f"   ✅ GOOD ITEM FOUND: Item {item_id} - {item.get('itemName', 'Unknown')}")
                    print(f"      Notes: {item.get('notes', 'No notes')}")
                elif item_id == 2:  # Damaged item (should not be here)
                    damaged_items_found.append(item)
                    print(f"   ❌ DAMAGED ITEM FOUND (ERROR): Item {item_id} - {item.get('itemName', 'Unknown')}")
                    print(f"      Notes: {item.get('notes', 'No notes')}")
        
        # Verify results
        if len(return_items_found) == 1 and len(damaged_items_found) == 0:
            print(f"\n🎉 SUCCESS! Filtering works correctly:")
            print(f"   ✅ 1 good condition item appears in picker dashboard")
            print(f"   ✅ 0 damaged items appear in picker dashboard")
            print(f"   📈 Items increased from {initial_count} to {final_count} (+{final_count - initial_count})")
            return True
        else:
            print(f"\n❌ FAILED! Filtering not working correctly:")
            print(f"   Expected: 1 good item, 0 damaged items")
            print(f"   Found: {len(return_items_found)} good items, {len(damaged_items_found)} damaged items")
            return False
    else:
        print(f"⚠️ Failed to get picker dashboard after approval: {picker_response_after.status_code}")
        return False

if __name__ == "__main__":
    success = test_damaged_items_filtering()
    if success:
        print("\n✅ Test PASSED - Damaged items are correctly filtered out!")
    else:
        print("\n❌ Test FAILED - Damaged items are still appearing in picker dashboard!")
