import requests

# Get token
token = requests.post('http://localhost:8002/api/v1/auth/token', data={'username': 'manager', 'password': 'manager123'}).json()['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Get orders
response = requests.get('http://localhost:8002/api/v1/orders', headers=headers)
orders = response.json()

if orders:
    first_order = orders[0]
    order_id = first_order.get('orderID')
    print(type(order_id))
    print(f"First order ID: {order_id}")
    print(f"Order status: {first_order.get('order_status')}")
    
    # Test regular orders endpoint
    response = requests.put(f'http://localhost:8002/api/v1/orders/{order_id}/status', 
                          headers=headers, 
                          params={'new_status': 'confirmed'})
    print(f"Regular endpoint status: {response.status_code}")
    
    # Test role-based endpoint
    response = requests.put(f'http://localhost:8002/api/v1/role-based/orders/{order_id}/status', 
                          headers=headers, 
                          params={'new_status': 'confirmed'})
    print(f"Role-based endpoint status: {response.status_code}")
    if response.status_code != 200:
        print(f"Role-based error: {response.text}")
else:
    print("No orders found")
