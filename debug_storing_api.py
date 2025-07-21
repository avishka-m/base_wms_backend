#!/usr/bin/env python3
"""
Debug script to check if the storing API integration is working
"""

import requests
import json

BASE_URL = "http://localhost:8002/api/v1"

def test_storing_api():
    """Test the storing API endpoints"""
    
    print("🔧 Testing Storing API Integration")
    print("=" * 50)
    
    # Login first
    login_data = {"username": "manager", "password": "manager123"}
    
    try:
        login_response = requests.post(f"{BASE_URL}/auth/token", data=login_data)
        if login_response.status_code == 200:
            token = login_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            print("✅ Login successful")
        else:
            print(f"❌ Login failed: {login_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # Test 1: Check if storing endpoint exists
    print("\n1️⃣ Testing storing endpoint...")
    try:
        storing_response = requests.get(f"{BASE_URL}/storing", headers=headers)
        print(f"Storing endpoint status: {storing_response.status_code}")
        if storing_response.status_code == 200:
            storing_jobs = storing_response.json()
            print(f"✅ Found {len(storing_jobs)} storing jobs")
        else:
            print(f"❌ Storing endpoint error: {storing_response.text}")
    except Exception as e:
        print(f"❌ Storing endpoint error: {e}")
    
    # Test 2: Check returns endpoint
    print("\n2️⃣ Testing returns endpoint...")
    try:
        returns_response = requests.get(f"{BASE_URL}/returns", headers=headers)
        print(f"Returns endpoint status: {returns_response.status_code}")
        if returns_response.status_code == 200:
            returns = returns_response.json()
            print(f"✅ Found {len(returns)} returns")
            
            # Look for a pending return to test with
            pending_returns = [r for r in returns if r.get("status") == "pending"]
            if pending_returns:
                test_return = pending_returns[0]
                return_id = test_return["returnID"]
                print(f"🎯 Found pending return {return_id} for testing")
                
                # Test 3: Try to approve this return
                print(f"\n3️⃣ Testing return approval for return {return_id}...")
                approval_data = {
                    "status": "approved",
                    "manager_notes": "Testing storing job creation"
                }
                
                try:
                    approve_response = requests.put(
                        f"{BASE_URL}/returns/{return_id}", 
                        json=approval_data, 
                        headers=headers
                    )
                    print(f"Approval response status: {approve_response.status_code}")
                    if approve_response.status_code == 200:
                        result = approve_response.json()
                        print("✅ Return approved successfully!")
                        print(f"Response keys: {list(result.keys())}")
                        
                        if "storing_job_created" in result:
                            print("🎉 Storing job creation info found in response!")
                            storing_info = result["storing_job_created"]
                            print(f"   Storing Job ID: {storing_info.get('storing_job_id')}")
                            print(f"   Assigned Worker: {storing_info.get('assigned_worker_id')}")
                        else:
                            print("⚠️ No storing_job_created field in response")
                            print(f"Full response: {json.dumps(result, indent=2)}")
                    else:
                        print(f"❌ Approval failed: {approve_response.text}")
                        
                except Exception as e:
                    print(f"❌ Approval error: {e}")
            else:
                print("⚠️ No pending returns found to test approval")
        else:
            print(f"❌ Returns endpoint error: {returns_response.text}")
    except Exception as e:
        print(f"❌ Returns endpoint error: {e}")
    
    # Test 4: Check storing jobs after approval
    print("\n4️⃣ Checking storing jobs after approval...")
    try:
        storing_response = requests.get(f"{BASE_URL}/storing", headers=headers)
        if storing_response.status_code == 200:
            storing_jobs = storing_response.json()
            print(f"✅ Found {len(storing_jobs)} storing jobs total")
            
            return_jobs = [job for job in storing_jobs if job.get("task_type") == "return_putaway"]
            print(f"📦 Found {len(return_jobs)} return putaway jobs")
            
            for job in return_jobs:
                print(f"   Job {job.get('storingID')}: Status={job.get('status')}, Worker={job.get('assignedWorkerID')}")
        else:
            print(f"❌ Storing jobs check failed: {storing_response.text}")
    except Exception as e:
        print(f"❌ Storing jobs error: {e}")

if __name__ == "__main__":
    test_storing_api()
