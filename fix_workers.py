#!/usr/bin/env python3

from app.utils.database import get_collection
from bson import ObjectId

def fix_worker_data():
    """Fix the remaining worker data issues"""
    workers = get_collection('workers')
    
    print('Checking and fixing worker data...')
    
    # Get all workers
    all_workers = list(workers.find())
    
    # Find the highest workerID
    max_worker_id = 0
    for worker in all_workers:
        if 'workerID' in worker:
            max_worker_id = max(max_worker_id, worker['workerID'])
    
    updates_made = 0
    
    for worker in all_workers:
        update_data = {}
        worker_id = worker['_id']
        
        print(f"Checking worker {worker_id}...")
        print(f"  Current fields: {list(worker.keys())}")
        
        # Fix name field - if has full_name but no name, copy full_name to name
        if 'name' not in worker and 'full_name' in worker:
            update_data['name'] = worker['full_name']
            print(f"  Adding 'name' field: {worker['full_name']}")
        
        # Fix workerID field - if missing workerID, add it
        if 'workerID' not in worker:
            max_worker_id += 1
            update_data['workerID'] = max_worker_id
            print(f"  Adding 'workerID': {max_worker_id}")
        
        # Apply updates
        if update_data:
            result = workers.update_one(
                {'_id': worker_id},
                {'$set': update_data}
            )
            if result.modified_count > 0:
                updates_made += 1
                print(f"  ✓ Updated worker {worker_id}")
            else:
                print(f"  ✗ Failed to update worker {worker_id}")
        else:
            print(f"  ✓ Worker {worker_id} already has correct fields")
    
    print(f'\nFixed {updates_made} workers.')
    
    # Verify all workers now have required fields
    print('\nVerifying all workers have required fields...')
    missing_fields = []
    
    for worker in workers.find():
        worker_id = worker['_id']
        if 'name' not in worker:
            missing_fields.append(f"Worker {worker_id} missing 'name'")
        if 'workerID' not in worker:
            missing_fields.append(f"Worker {worker_id} missing 'workerID'")
    
    if missing_fields:
        print("❌ Issues found:")
        for issue in missing_fields:
            print(f"  - {issue}")
    else:
        print("✅ All workers have required fields!")

if __name__ == "__main__":
    fix_worker_data()
