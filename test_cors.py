import requests
import json

def test_cors():
    print("Testing CORS headers...")
    
    try:
        # 1. Login to get token
        print("1. Testing authentication...")
        auth_response = requests.post('http://localhost:8002/api/v1/auth/token', 
                                     data={'username': 'manager', 'password': 'manager123'})
        
        print(f"Auth Status: {auth_response.status_code}")
        
        if auth_response.status_code == 200:
            token = auth_response.json()['access_token']
            
            # 2. Test workers endpoint with authentication
            print("2. Testing authenticated workers request...")
            headers = {'Authorization': f'Bearer {token}'}
            workers_response = requests.get('http://localhost:8002/api/v1/workers/', headers=headers)
            
            print(f"Workers Status: {workers_response.status_code}")
            if workers_response.status_code == 200:
                print(f"Workers Count: {len(workers_response.json())}")
            
            # Check CORS headers in successful response
            cors_headers = {
                'Access-Control-Allow-Origin': workers_response.headers.get('Access-Control-Allow-Origin', 'NOT PRESENT'),
                'Access-Control-Allow-Methods': workers_response.headers.get('Access-Control-Allow-Methods', 'NOT PRESENT'),
                'Access-Control-Allow-Headers': workers_response.headers.get('Access-Control-Allow-Headers', 'NOT PRESENT')
            }
            
            print("Authenticated Request CORS Headers:")
            for header, value in cors_headers.items():
                print(f"  {header}: {value}")
            
            # 3. Test unauthenticated request to see if it returns proper CORS headers
            print("\n3. Testing unauthenticated request...")
            unauth_response = requests.get('http://localhost:8002/api/v1/workers/')
            
            print(f"Unauth Status: {unauth_response.status_code}")
            
            unauth_cors = {
                'Access-Control-Allow-Origin': unauth_response.headers.get('Access-Control-Allow-Origin', 'NOT PRESENT'),
                'Access-Control-Allow-Methods': unauth_response.headers.get('Access-Control-Allow-Methods', 'NOT PRESENT'),
                'Access-Control-Allow-Headers': unauth_response.headers.get('Access-Control-Allow-Headers', 'NOT PRESENT')
            }
            
            print("Unauthenticated Request CORS Headers:")
            for header, value in unauth_cors.items():
                print(f"  {header}: {value}")
                
            # Check if both requests have CORS headers
            auth_has_cors = cors_headers['Access-Control-Allow-Origin'] != 'NOT PRESENT'
            unauth_has_cors = unauth_cors['Access-Control-Allow-Origin'] != 'NOT PRESENT'
            
            print(f"\nResults:")
            print(f"✅ Authenticated request has CORS headers: {auth_has_cors}")
            print(f"✅ Unauthenticated request has CORS headers: {unauth_has_cors}")
            
            if auth_has_cors and unauth_has_cors:
                print("\n🎉 CORS is working correctly for both authenticated and unauthenticated requests!")
                return True
            else:
                print("\n❌ CORS headers are missing from some requests")
                return False
                
        else:
            print(f"Auth failed: {auth_response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cors()
