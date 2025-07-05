import requests

print("Testing workflow endpoints...")

# First, let's get a token for authentication
auth_response = requests.post('http://localhost:8002/api/v1/auth/token', {
    'username': 'manager',
    'password': 'manager123'
})

if auth_response.status_code == 200:
    token = auth_response.json()['access_token']
    headers = {'Authorization': f'Bearer {token}'}
    print("✓ Successfully authenticated")
else:
    print("✗ Authentication failed")
    headers = {}

# Test GET endpoints
get_endpoints = [
    '/api/v1/workflow/status/overview', 
    '/api/v1/workflow/active-tasks'
]

for endpoint in get_endpoints:
    try:
        response = requests.get(f'http://localhost:8002{endpoint}', headers=headers, timeout=5)
        print(f'{endpoint}: {response.status_code}')
        if response.status_code == 200:
            print(f'  ✓ Success: {response.json()["success"]}')
        elif response.status_code != 404:
            print(f'  Response: {response.text[:100]}...')
    except Exception as e:
        print(f'{endpoint}: Error - {e}')

# Test POST endpoint
post_endpoint = '/api/v1/workflow/optimization/analyze'
try:
    response = requests.post(f'http://localhost:8002{post_endpoint}', 
                           headers=headers, 
                           json={'worker_roles': ['picker', 'packer']}, 
                           timeout=5)
    print(f'{post_endpoint}: {response.status_code}')
    if response.status_code == 200:
        print(f'  ✓ Success: {response.json()["success"]}')
    elif response.status_code != 404:
        print(f'  Response: {response.text[:100]}...')
except Exception as e:
    print(f'{post_endpoint}: Error - {e}')

print("Done testing workflow endpoints.")
