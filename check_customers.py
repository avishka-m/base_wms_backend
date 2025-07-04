#!/usr/bin/env python3

from app.utils.database import get_collection

def check_customers():
    """Check current customers in the database"""
    customers = get_collection('customers')
    print('Current customers in database:')
    
    customer_list = list(customers.find())
    if not customer_list:
        print('No customers found in database')
        return
    
    for customer in customer_list:
        print(f'ID: {customer.get("customerID", "N/A")}')
        print(f'Name: {customer.get("name", "N/A")}')
        print(f'Email: {customer.get("email", "N/A")}')
        print(f'Fields: {list(customer.keys())}')
        print('-' * 50)

if __name__ == "__main__":
    check_customers()
