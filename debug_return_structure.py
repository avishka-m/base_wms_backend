#!/usr/bin/env python3
"""
Debug script to manually test the storing job creation function
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8002/api/v1"

def debug_return_structure():
    """Debug the structure of returns to understand the issue"""
    
    print("🔍 Debugging Return Structure and Storing Job Creation")
    print("=" * 60)
    
    # Login first
    login_data = {"username": "manager", "password": "manager123"}
    
    try:
        login_response = requests.post(f"{BASE_URL}/auth/token", data=login_data)
        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("✅ Login successful")
        else:
            print(f"❌ Login failed")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # Get a specific return to examine its structure
    print("\n1️⃣ Examining return structure...")
    try:
        returns_response = requests.get(f"{BASE_URL}/returns", headers=headers)
        if returns_response.status_code == 200:
            returns = returns_response.json()
            
            # Find a pending return
            pending_return = None
            for return_record in returns:
                if return_record.get("status") == "pending":
                    pending_return = return_record
                    break
            
            if pending_return:
                print(f"📋 Found pending return {pending_return['returnID']}")
                print(f"Return structure:")
                print(json.dumps(pending_return, indent=2, default=str))
                
                # Check items structure
                items = pending_return.get("items", [])
                print(f"\n📦 Items in return ({len(items)} items):")
                for i, item in enumerate(items):
                    print(f"  Item {i+1}:")
                    print(f"    itemID: {item.get('itemID')}")
                    print(f"    quantity: {item.get('quantity')}")
                    print(f"    locationID: {item.get('locationID')}")
                    print(f"    reason: {item.get('reason')}")
                    print(f"    condition: {item.get('condition')}")
                
                # Test manual storing job creation with this return
                print(f"\n2️⃣ Testing manual storing job creation...")
                test_storing_job_creation(pending_return, headers)
                
            else:
                print("⚠️ No pending returns found")
                
        else:
            print(f"❌ Failed to get returns: {returns_response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_storing_job_creation(return_record, headers):
    """Manually test storing job creation"""
    
    print(f"Creating storing job for return {return_record['returnID']}...")
    
    # Check if workers exist
    try:
        workers_response = requests.get(f"{BASE_URL}/workers", headers=headers)
        if workers_response.status_code == 200:
            workers = workers_response.json()
            pickers = [w for w in workers if w.get("role") == "Picker"]
            print(f"📋 Found {len(pickers)} pickers available")
            for picker in pickers:
                print(f"  Picker ID: {picker.get('workerID')}, Username: {picker.get('username')}")
        else:
            print(f"⚠️ Could not get workers: {workers_response.status_code}")
    except Exception as e:
        print(f"⚠️ Worker check error: {e}")
    
    # Try to create a storing job manually using the storing API
    print(f"\n3️⃣ Creating storing job via storing API...")
    
    try:
        # Prepare storing items from return items
        storing_items = []
        for item in return_record.get("items", []):
            storing_item = {
                "itemID": item.get("itemID"),
                "returnID": return_record.get("returnID"),
                "locationID": item.get("locationID") or "B01.1",  # Default location if not specified
                "quantity": item.get("quantity", 1),
                "reason": f"Return approved - {item.get('reason', 'Customer return')}"
            }
            storing_items.append(storing_item)
            print(f"  📦 Storing item: {storing_item}")
        
        # Create storing job payload
        storing_job_data = {
            "assignedWorkerID": 5001,  # Use first picker or default
            "priority": 2,  # Medium priority for returns
            "status": "pending",
            "task_type": "return_putaway",
            "items": storing_items
        }
        
        print(f"\n📋 Storing job payload:")
        print(json.dumps(storing_job_data, indent=2))
        
        # Send to storing API
        storing_response = requests.post(
            f"{BASE_URL}/storing", 
            json=storing_job_data, 
            headers=headers
        )
        
        print(f"\n📤 Storing API response: {storing_response.status_code}")
        if storing_response.status_code == 200 or storing_response.status_code == 201:
            result = storing_response.json()
            print(f"✅ Storing job created successfully!")
            print(f"   Storing Job ID: {result.get('storingID')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Assigned Worker: {result.get('assignedWorkerID')}")
        else:
            print(f"❌ Storing job creation failed:")
            print(f"   Error: {storing_response.text}")
            
    except Exception as e:
        print(f"❌ Manual storing job creation error: {e}")

if __name__ == "__main__":
    debug_return_structure()
