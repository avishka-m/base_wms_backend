#!/usr/bin/env python3

from app.utils.database import get_collection
import json

def check_workers():
    workers = get_collection('workers')
    print('Current workers in database:')
    
    for worker in workers.find():
        print(f'ID: {worker.get("_id")}')
        print(f'Fields: {list(worker.keys())}')
        print(f'Data: {worker}')
        print('-' * 50)

if __name__ == "__main__":
    check_workers()
