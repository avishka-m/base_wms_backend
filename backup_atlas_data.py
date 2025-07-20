#!/usr/bin/env python3
"""
MongoDB Atlas Data Backup Script

This script creates a complete backup of your MongoDB Atlas database
before migrating to a new region. It exports all collections and data
in multiple formats for maximum safety.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import zipfile

# Add config path
config_path = os.path.join(os.path.dirname(__file__), 'config')
sys.path.insert(0, config_path)

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from bson import json_util
import pandas as pd

# Import optimized connection
from config.atlas_optimization import get_database_url, get_client_options

class AtlasDataBackup:
    """Complete backup solution for MongoDB Atlas data migration."""
    
    def __init__(self):
        self.mongodb_url = get_database_url()
        self.database_name = os.getenv('DATABASE_NAME', 'warehouse_management')
        self.backup_dir = f"atlas_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.client_options = get_client_options()
        
        # Create backup directory
        os.makedirs(self.backup_dir, exist_ok=True)
        print(f"📁 Backup directory created: {self.backup_dir}")
    
    def test_connection(self) -> bool:
        """Test connection to the current Atlas cluster."""
        try:
            client = MongoClient(self.mongodb_url, **self.client_options)
            db = client[self.database_name]
            db.command('ping')
            client.close()
            print("✅ Successfully connected to Atlas cluster")
            return True
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to Atlas: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and collection information."""
        try:
            client = MongoClient(self.mongodb_url, **self.client_options)
            db = client[self.database_name]
            
            # Get database stats
            db_stats = db.command('dbStats')
            collections = db.list_collection_names()
            
            collection_stats = {}
            total_documents = 0
            
            for collection_name in collections:
                collection = db[collection_name]
                count = collection.count_documents({})
                collection_stats[collection_name] = {
                    'document_count': count,
                    'indexes': list(collection.list_indexes())
                }
                total_documents += count
            
            stats = {
                'database_name': self.database_name,
                'total_collections': len(collections),
                'total_documents': total_documents,
                'database_size': db_stats.get('dataSize', 0),
                'collections': collection_stats,
                'backup_timestamp': datetime.now().isoformat()
            }
            
            client.close()
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get database stats: {e}")
            return {}
    
    def backup_using_mongodump(self) -> bool:
        """Backup using mongodump (most reliable method)."""
        print("\n🔄 Method 1: Creating backup using mongodump...")
        
        try:
            dump_dir = os.path.join(self.backup_dir, "mongodump")
            os.makedirs(dump_dir, exist_ok=True)
            
            # mongodump command
            cmd = [
                "mongodump",
                "--uri", self.mongodb_url,
                "--out", dump_dir
            ]
            
            print(f"   Running: mongodump --uri [HIDDEN] --out {dump_dir}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("   ✅ mongodump backup completed successfully")
                return True
            else:
                print(f"   ❌ mongodump failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ mongodump timed out (database too large)")
            return False
        except FileNotFoundError:
            print("   ⚠️  mongodump not found. Install MongoDB tools: https://www.mongodb.com/try/download/database-tools")
            return False
        except Exception as e:
            print(f"   ❌ mongodump error: {e}")
            return False
    
    def backup_using_pymongo(self) -> bool:
        """Backup using PyMongo (Python-based backup)."""
        print("\n🔄 Method 2: Creating backup using PyMongo...")
        
        try:
            client = MongoClient(self.mongodb_url, **self.client_options)
            db = client[self.database_name]
            
            pymongo_dir = os.path.join(self.backup_dir, "pymongo_backup")
            os.makedirs(pymongo_dir, exist_ok=True)
            
            collections = db.list_collection_names()
            backed_up_collections = 0
            
            for collection_name in collections:
                try:
                    print(f"   📦 Backing up collection: {collection_name}")
                    collection = db[collection_name]
                    
                    # Get all documents
                    documents = list(collection.find())
                    
                    if documents:
                        # Save as JSON
                        json_file = os.path.join(pymongo_dir, f"{collection_name}.json")
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(documents, f, default=json_util.default, indent=2, ensure_ascii=False)
                        
                        print(f"   ✅ {collection_name}: {len(documents)} documents saved")
                        backed_up_collections += 1
                    else:
                        print(f"   📝 {collection_name}: Empty collection")
                        backed_up_collections += 1
                        
                except Exception as e:
                    print(f"   ❌ Failed to backup {collection_name}: {e}")
            
            client.close()
            
            if backed_up_collections > 0:
                print(f"   ✅ PyMongo backup completed: {backed_up_collections} collections")
                return True
            else:
                print("   ❌ No collections were backed up")
                return False
                
        except Exception as e:
            print(f"   ❌ PyMongo backup error: {e}")
            return False
    
    def backup_using_csv_export(self) -> bool:
        """Backup critical collections as CSV (Excel-readable)."""
        print("\n🔄 Method 3: Creating CSV exports for critical data...")
        
        try:
            client = MongoClient(self.mongodb_url, **self.client_options)
            db = client[self.database_name]
            
            csv_dir = os.path.join(self.backup_dir, "csv_exports")
            os.makedirs(csv_dir, exist_ok=True)
            
            # Critical collections to export as CSV
            critical_collections = [
                'inventory', 'orders', 'customers', 'workers', 
                'suppliers', 'locations', 'warehouses'
            ]
            
            exported_count = 0
            
            for collection_name in critical_collections:
                try:
                    if collection_name in db.list_collection_names():
                        print(f"   📊 Exporting {collection_name} to CSV...")
                        collection = db[collection_name]
                        
                        # Get all documents
                        documents = list(collection.find())
                        
                        if documents:
                            # Convert to DataFrame and save as CSV
                            # Flatten nested objects for CSV compatibility
                            flattened_docs = []
                            for doc in documents:
                                flat_doc = self._flatten_dict(doc)
                                flattened_docs.append(flat_doc)
                            
                            df = pd.DataFrame(flattened_docs)
                            csv_file = os.path.join(csv_dir, f"{collection_name}.csv")
                            df.to_csv(csv_file, index=False, encoding='utf-8')
                            
                            print(f"   ✅ {collection_name}.csv: {len(documents)} records")
                            exported_count += 1
                        else:
                            print(f"   📝 {collection_name}: Empty collection")
                            
                except Exception as e:
                    print(f"   ❌ Failed to export {collection_name}: {e}")
            
            client.close()
            
            if exported_count > 0:
                print(f"   ✅ CSV export completed: {exported_count} files")
                return True
            else:
                print("   ⚠️  No CSV files were created")
                return False
                
        except Exception as e:
            print(f"   ❌ CSV export error: {e}")
            return False
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # Handle list of dictionaries (like order items)
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", str(item)))
            else:
                # Convert ObjectId and other non-serializable types to string
                if hasattr(v, '__str__'):
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
        return dict(items)
    
    def create_restore_script(self, stats: Dict[str, Any]) -> None:
        """Create a script to restore data to the new cluster."""
        print("\n📝 Creating restore script...")
        
        restore_script = f'''#!/usr/bin/env python3
"""
MongoDB Atlas Data Restore Script
Auto-generated on {datetime.now().isoformat()}

This script restores data from the backup to your new Singapore Atlas cluster.
"""

import os
import sys
import json
import subprocess
from pymongo import MongoClient
from bson import json_util

def restore_using_mongorestore(new_mongodb_url, database_name):
    """Restore using mongorestore (recommended method)."""
    print("🔄 Restoring data using mongorestore...")
    
    dump_dir = os.path.join("{self.backup_dir}", "mongodump")
    if not os.path.exists(dump_dir):
        print("❌ mongodump backup not found")
        return False
    
    try:
        cmd = [
            "mongorestore",
            "--uri", new_mongodb_url,
            "--dir", dump_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ mongorestore completed successfully")
            return True
        else:
            print(f"❌ mongorestore failed: {{result.stderr}}")
            return False
            
    except Exception as e:
        print(f"❌ mongorestore error: {{e}}")
        return False

def restore_using_pymongo(new_mongodb_url, database_name):
    """Restore using PyMongo as fallback."""
    print("🔄 Restoring data using PyMongo...")
    
    pymongo_dir = os.path.join("{self.backup_dir}", "pymongo_backup")
    if not os.path.exists(pymongo_dir):
        print("❌ PyMongo backup not found")
        return False
    
    try:
        client = MongoClient(new_mongodb_url)
        db = client[database_name]
        
        json_files = [f for f in os.listdir(pymongo_dir) if f.endswith('.json')]
        restored_collections = 0
        
        for json_file in json_files:
            collection_name = json_file.replace('.json', '')
            
            with open(os.path.join(pymongo_dir, json_file), 'r', encoding='utf-8') as f:
                documents = json.load(f, object_hook=json_util.object_hook)
            
            if documents:
                # Clear existing collection (optional)
                db[collection_name].delete_many({{}})
                
                # Insert documents
                db[collection_name].insert_many(documents)
                print(f"✅ {{collection_name}}: {{len(documents)}} documents restored")
                restored_collections += 1
            else:
                print(f"📝 {{collection_name}}: Empty collection")
        
        client.close()
        
        if restored_collections > 0:
            print(f"✅ PyMongo restore completed: {{restored_collections}} collections")
            return True
        else:
            print("❌ No collections were restored")
            return False
            
    except Exception as e:
        print(f"❌ PyMongo restore error: {{e}}")
        return False

if __name__ == "__main__":
    print("MongoDB Atlas Data Restore")
    print("=" * 40)
    
    # Get new cluster connection details
    new_mongodb_url = input("Enter your NEW Singapore cluster connection string: ")
    database_name = "{self.database_name}"
    
    print(f"Database: {{database_name}}")
    print(f"Backup date: {stats.get('backup_timestamp', 'Unknown')}")
    print(f"Collections to restore: {stats.get('total_collections', 'Unknown')}")
    print(f"Total documents: {stats.get('total_documents', 'Unknown')}")
    print()
    
    # Try mongorestore first (most reliable)
    if restore_using_mongorestore(new_mongodb_url, database_name):
        print("🎉 Data restoration completed successfully!")
    else:
        print("⚠️  mongorestore failed, trying PyMongo method...")
        if restore_using_pymongo(new_mongodb_url, database_name):
            print("🎉 Data restoration completed using PyMongo!")
        else:
            print("❌ All restore methods failed. Check the CSV exports for manual import.")
'''
        
        script_file = os.path.join(self.backup_dir, "restore_data.py")
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(restore_script)
        
        print(f"   ✅ Restore script created: {script_file}")
    
    def create_backup_summary(self, stats: Dict[str, Any], methods_success: Dict[str, bool]) -> None:
        """Create a summary of the backup process."""
        print("\n📋 Creating backup summary...")
        
        summary = {
            'backup_info': {
                'timestamp': datetime.now().isoformat(),
                'database_name': self.database_name,
                'backup_directory': self.backup_dir,
                'total_collections': stats.get('total_collections', 0),
                'total_documents': stats.get('total_documents', 0),
                'database_size_bytes': stats.get('database_size', 0)
            },
            'backup_methods': methods_success,
            'collections': stats.get('collections', {}),
            'next_steps': [
                "1. Create new Atlas cluster in Singapore (ap-southeast-1)",
                "2. Get the new cluster connection string", 
                "3. Run the restore_data.py script",
                "4. Update your .env file with new connection details",
                "5. Test your application with the new cluster"
            ]
        }
        
        summary_file = os.path.join(self.backup_dir, "backup_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   ✅ Backup summary created: {summary_file}")
    
    def create_zip_archive(self) -> None:
        """Create a ZIP archive of the backup for easy transfer."""
        print("\n📦 Creating ZIP archive...")
        
        try:
            zip_file = f"{self.backup_dir}.zip"
            
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.backup_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.backup_dir)
                        zipf.write(file_path, arcname)
            
            zip_size = os.path.getsize(zip_file) / (1024 * 1024)  # MB
            print(f"   ✅ ZIP archive created: {zip_file} ({zip_size:.1f} MB)")
            
        except Exception as e:
            print(f"   ❌ Failed to create ZIP archive: {e}")
    
    def run_complete_backup(self) -> bool:
        """Run the complete backup process."""
        print("🚀 Starting Complete MongoDB Atlas Backup")
        print("=" * 50)
        
        # Test connection
        if not self.test_connection():
            return False
        
        # Get database stats
        print("\n📊 Analyzing database...")
        stats = self.get_database_stats()
        
        if not stats:
            print("❌ Failed to analyze database")
            return False
        
        print(f"   Database: {stats['database_name']}")
        print(f"   Collections: {stats['total_collections']}")
        print(f"   Total Documents: {stats['total_documents']:,}")
        print(f"   Database Size: {stats['database_size']:,} bytes")
        
        # Run backup methods
        methods_success = {
            'mongodump': self.backup_using_mongodump(),
            'pymongo': self.backup_using_pymongo(),
            'csv_export': self.backup_using_csv_export()
        }
        
        # Create restore script and summary
        self.create_restore_script(stats)
        self.create_backup_summary(stats, methods_success)
        self.create_zip_archive()
        
        # Final report
        print("\n🎯 BACKUP COMPLETE!")
        print("=" * 30)
        
        successful_methods = sum(methods_success.values())
        print(f"✅ Successful backup methods: {successful_methods}/3")
        
        for method, success in methods_success.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {method}: {status}")
        
        if successful_methods > 0:
            print(f"\n📁 Backup location: {self.backup_dir}")
            print(f"📦 ZIP archive: {self.backup_dir}.zip")
            print("\n🔄 Next steps:")
            print("   1. Create new Atlas cluster in Singapore (ap-southeast-1)")
            print("   2. Run: python restore_data.py")
            print("   3. Update your .env with new connection string")
            return True
        else:
            print("\n❌ All backup methods failed!")
            print("   Check your connection and try again")
            return False

def main():
    """Main function to run the backup."""
    backup = AtlasDataBackup()
    success = backup.run_complete_backup()
    
    if success:
        print("\n🎉 Backup completed successfully!")
        print("Your data is now safely backed up and ready for migration.")
    else:
        print("\n❌ Backup failed!")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    main() 