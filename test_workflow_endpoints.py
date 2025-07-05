import requests

def test_workflow_endpoints():
    print("Testing new workflow endpoints...")
    
    # Get auth token first
    try:
        auth_response = requests.post('http://localhost:8002/api/v1/auth/token', 
                                     data={'username': 'manager', 'password': 'manager123'})
        
        if auth_response.status_code != 200:
            print(f"❌ Auth failed: {auth_response.status_code}")
            return False
            
        token = auth_response.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        
        # Test the workflow endpoints
        endpoints = [
            ('POST', '/api/v1/workflow/optimization/analyze', {'worker_roles': ['Picker', 'Packer']}),
            ('GET', '/api/v1/workflow/status/overview', None),
            ('GET', '/api/v1/workflow/active-tasks', None)
        ]
        
        for method, endpoint, data in endpoints:
            try:
                print(f"\n--- Testing {method} {endpoint} ---")
                
                if method == 'POST':
                    response = requests.post(f'http://localhost:8002{endpoint}', 
                                           json=data, headers=headers)
                else:
                    response = requests.get(f'http://localhost:8002{endpoint}', 
                                          headers=headers)
                
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Success: {data.get('success', False)}")
                    if 'data' in data:
                        print(f"Data keys: {list(data['data'].keys()) if isinstance(data['data'], dict) else type(data['data'])}")
                else:
                    print(f"❌ Failed: {response.text[:200]}...")
                    
            except Exception as e:
                print(f"❌ Error testing {endpoint}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_workflow_endpoints()