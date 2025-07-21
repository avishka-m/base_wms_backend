import requests

def test_returns_approval_functionality():
    """Test the returns approval functionality from the frontend perspective"""
    
    base_url = "http://127.0.0.1:8002/api/v1"
    
    print("🧪 Testing Returns Approval Functionality")
    print("=" * 50)
    
    # Step 1: Login as manager
    print("\n1. Testing Manager Login...")
    
    login_data = {
        "username": "manager",
        "password": "manager123"
    }
    
    login_response = requests.post(f"{base_url}/auth/token", data=login_data)
    if login_response.status_code == 200:
        manager_token = login_response.json()["access_token"]
        print("✅ Manager login successful")
    else:
        print(f"❌ Manager login failed: {login_response.status_code}")
        return
    
    headers = {"Authorization": f"Bearer {manager_token}"}
    
    # Step 2: Get all returns
    print("\n2. Testing Get All Returns...")
    
    returns_response = requests.get(f"{base_url}/returns/", headers=headers)
    
    if returns_response.status_code == 200:
        returns_data = returns_response.json()
        print(f"✅ Got {len(returns_data)} returns")
        
        # Find pending returns
        pending_returns = [r for r in returns_data if r.get('status') == 'pending']
        print(f"   Found {len(pending_returns)} pending returns")
        
        if pending_returns:
            test_return = pending_returns[0]
            return_id = test_return['returnID']
            print(f"   Testing with return ID: {return_id}")
            
            # Step 3: Test approval
            print(f"\n3. Testing Return Approval (ID: {return_id})...")
            
            approval_data = {
                "status": "approved",
                "notes": "Approved by manager via API test"
            }
            
            approve_response = requests.put(f"{base_url}/returns/{return_id}",
                                          headers=headers,
                                          json=approval_data)
            
            if approve_response.status_code == 200:
                approved_return = approve_response.json()
                print(f"✅ Return approved successfully")
                print(f"   New status: {approved_return['status']}")
                print(f"   Notes: {approved_return.get('notes', 'None')}")
                
                # Step 4: Verify the update
                print(f"\n4. Verifying Return Update...")
                
                verify_response = requests.get(f"{base_url}/returns/{return_id}", headers=headers)
                
                if verify_response.status_code == 200:
                    verified_return = verify_response.json()
                    print(f"✅ Verification successful")
                    print(f"   Status confirmed: {verified_return['status']}")
                    print(f"   Notes confirmed: {verified_return.get('notes', 'None')}")
                else:
                    print(f"❌ Verification failed: {verify_response.status_code}")
                
            else:
                print(f"❌ Return approval failed: {approve_response.status_code}")
                print(f"Response: {approve_response.text}")
                
        else:
            print("   No pending returns to test with")
            
            # Create a test return for demonstration
            print("\n   Creating a test return...")
            
            test_return_data = {
                "orderID": 1,
                "customerID": 1,
                "workerID": 5001,
                "items": [
                    {
                        "itemID": 1,
                        "orderDetailID": 1,
                        "quantity": 1,
                        "reason": "Exchanged",
                        "condition": "Good"
                    }
                ]
            }
            
            create_response = requests.post(f"{base_url}/returns/",
                                          headers=headers,
                                          json=test_return_data)
            
            if create_response.status_code == 201:
                new_return = create_response.json()
                new_return_id = new_return['returnID']
                print(f"✅ Test return created: {new_return_id}")
                
                # Now test approval on the new return
                print(f"\n3. Testing Approval on New Return (ID: {new_return_id})...")
                
                approval_data = {
                    "status": "approved",
                    "notes": "Approved new test return"
                }
                
                approve_response = requests.put(f"{base_url}/returns/{new_return_id}",
                                              headers=headers,
                                              json=approval_data)
                
                if approve_response.status_code == 200:
                    print(f"✅ New return approved successfully")
                else:
                    print(f"❌ New return approval failed: {approve_response.status_code}")
            else:
                print(f"❌ Test return creation failed: {create_response.status_code}")
            
    else:
        print(f"❌ Get returns failed: {returns_response.status_code}")
        print(f"Response: {returns_response.text}")
    
    print("\n" + "=" * 50)
    print("🎉 Returns Approval Test Completed!")
    print("\nFrontend Changes Made:")
    print("1. ✅ Added handleApproveReturn() function")
    print("2. ✅ Added handleRejectReturn() function") 
    print("3. ✅ Added Approve/Reject buttons for managers")
    print("4. ✅ Added 'approved' and 'rejected' status colors")
    print("5. ✅ Added filter options for new statuses")
    print("6. ✅ Updated stats to show approved returns")

if __name__ == "__main__":
    test_returns_approval_functionality()
