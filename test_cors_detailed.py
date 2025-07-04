import requests

def test_cors_headers():
    print("Testing CORS headers on different endpoints...")
    
    endpoints = [
        ('/', 'Root endpoint'),
        ('/health', 'Health check endpoint'),
        ('/api/v1/workers/', 'Workers endpoint (unauthenticated)'),
    ]
    
    for endpoint, description in endpoints:
        try:
            print(f"\n--- Testing {description} ---")
            response = requests.get(f'http://localhost:8002{endpoint}')
            
            print(f"Status: {response.status_code}")
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers',
                'Access-Control-Allow-Credentials'
            ]
            
            print("CORS Headers:")
            for header in cors_headers:
                value = response.headers.get(header, 'NOT PRESENT')
                print(f"  {header}: {value}")
            
            # Check for any other Access-Control headers
            other_cors = {k: v for k, v in response.headers.items() if k.startswith('Access-Control')}
            if other_cors:
                print("Other CORS Headers:")
                for k, v in other_cors.items():
                    print(f"  {k}: {v}")
            
        except Exception as e:
            print(f"Error testing {endpoint}: {e}")

if __name__ == "__main__":
    test_cors_headers()
