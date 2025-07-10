#!/usr/bin/env python3

from app.utils.database import get_collection
import json

def migrate_workers():
    """
    Migrate worker records to ensure consistent field names
    - Convert 'full_name' to 'name' if 'name' doesn't exist
    - Ensure all workers have 'workerID'
    """
    workers = get_collection('workers')
    
    print('Migrating workers to consistent format...')
    
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
        
        # Fix name field
        if 'name' not in worker and 'full_name' in worker:
            update_data['name'] = worker['full_name']
            print(f"Worker {worker['_id']}: Adding 'name' field from 'full_name'")
        
        # Fix workerID field
        if 'workerID' not in worker:
            max_worker_id += 1
            update_data['workerID'] = max_worker_id
            print(f"Worker {worker['_id']}: Adding 'workerID' = {max_worker_id}")
        
        # Apply updates
        if update_data:
            workers.update_one(
                {'_id': worker['_id']},
                {'$set': update_data}
            )
            updates_made += 1
            print(f"Updated worker {worker['_id']}")
    
    print(f'\nMigration complete! Updated {updates_made} workers.')
    
    # Verify the migration
    print('\nVerifying migration...')
    for worker in workers.find():
        if 'name' not in worker:
            print(f"ERROR: Worker {worker['_id']} still missing 'name' field")
        if 'workerID' not in worker:
            print(f"ERROR: Worker {worker['_id']} still missing 'workerID' field")
    
    print('Migration verification complete!')

if __name__ == "__main__":
    migrate_workers()
