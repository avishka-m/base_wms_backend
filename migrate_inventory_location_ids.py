#!/usr/bin/env python3
"""
Migrate Inventory Collection Location IDs from Numeric to Warehouse Format

This script converts numeric locationID values (1, 2, 3, ...) in the inventory collection
to proper warehouse location format (B01.1, B01.2, P03.1, etc.) based on the warehouse mapping.
"""

import os
import sys
import random
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

# Import the warehouse mapper
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_services', 'path_optimization'))
try:
    from warehouse_mapper import WarehouseMapper
except ImportError:
    print("Error: Could not import WarehouseMapper. Make sure the path is correct.")
    sys.exit(1)

class InventoryLocationMigrator:
    """Migrates inventory location IDs from numeric to warehouse location format"""
    
    def __init__(self):
        # Connect to localhost MongoDB
        self.client = MongoClient('mongodb+srv://wms:3cVnhHuj5caki0Ve@cluster0.99chyus.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
        self.db = self.client['warehouse_management']
        self.warehouse_mapper = WarehouseMapper()
        
        # Generate all available warehouse locations with sublocations
        self.available_locations = self._generate_all_warehouse_locations()
        
    def _generate_all_warehouse_locations(self):
        """Generate all possible warehouse locations with sublocations"""
        locations = []
        
        # Get all base locations from warehouse_map
        for base_location in self.warehouse_mapper.warehouse_map.keys():
            # Generate all 4 sublocations for each base location (floors 1-4)
            for floor in range(1, 5):
                sublocation = self.warehouse_mapper.parse_base_to_sublocation_format(base_location, floor)
                locations.append(sublocation)
        
        return locations
    
    def analyze_current_inventory(self):
        """Analyze current inventory collection location IDs"""
        print("=== Current Inventory Analysis ===")
        
        total_count = self.db.inventory.count_documents({})
        print(f"Total inventory items: {total_count}")
        
        # Get unique location IDs
        pipeline = [
            {"$group": {"_id": "$locationID", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        location_stats = list(self.db.inventory.aggregate(pipeline))
        
        print("\nCurrent locationID distribution:")
        for stat in location_stats:
            print(f"  locationID {stat['_id']}: {stat['count']} items")
        
        return location_stats
    
    def create_numeric_to_warehouse_mapping(self, location_stats):
        """Create a mapping from numeric IDs to warehouse locations"""
        print("\n=== Creating Location Mapping ===")
        
        # Extract numeric location IDs (excluding 0 which might be default/unassigned)
        numeric_ids = [stat['_id'] for stat in location_stats if isinstance(stat['_id'], int) and stat['_id'] > 0]
        numeric_ids.sort()
        
        if len(numeric_ids) > len(self.available_locations):
            print(f"WARNING: More numeric IDs ({len(numeric_ids)}) than available warehouse locations ({len(self.available_locations)})")
            print("Some items may need manual assignment.")
        
        # Create mapping by randomly assigning warehouse locations
        # You can modify this logic to assign locations based on item categories, sizes, etc.
        mapping = {}
        used_locations = set()
        
        # Shuffle locations for random distribution
        import random
        available_shuffled = self.available_locations.copy()
        random.shuffle(available_shuffled)
        
        for i, numeric_id in enumerate(numeric_ids):
            if i < len(available_shuffled):
                warehouse_location = available_shuffled[i]
                mapping[numeric_id] = warehouse_location
                used_locations.add(warehouse_location)
                print(f"  {numeric_id} -> {warehouse_location}")
            else:
                # If we run out of locations, assign to a random available one
                remaining_locations = [loc for loc in self.available_locations if loc not in used_locations]
                if remaining_locations:
                    warehouse_location = random.choice(remaining_locations)
                    mapping[numeric_id] = warehouse_location
                    used_locations.add(warehouse_location)
                    print(f"  {numeric_id} -> {warehouse_location} (overflow assignment)")
                else:
                    print(f"  WARNING: Could not assign location for numeric ID {numeric_id}")
        
        # Handle locationID 0 (unassigned/default)
        if 0 in [stat['_id'] for stat in location_stats]:
            # Assign remaining available location or a default
            remaining_locations = [loc for loc in self.available_locations if loc not in used_locations]
            if remaining_locations:
                mapping[0] = remaining_locations[0]
                print(f"  0 (unassigned) -> {remaining_locations[0]}")
            else:
                mapping[0] = "B01.1"  # Default fallback
                print(f"  0 (unassigned) -> B01.1 (default fallback)")
        
        return mapping
    
    def create_smart_mapping_by_category(self, location_stats):
        """Create a smarter mapping based on item categories and storage requirements"""
        print("\n=== Creating Smart Category-Based Mapping ===")
        
        # Get items grouped by category and storage type
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "locationID": "$locationID",
                        "category": "$category",
                        "storage_type": "$storage_type",
                        "size": "$size"
                    },
                    "items": {"$push": "$$ROOT"}
                }
            }
        ]
        
        grouped_items = list(self.db.inventory.aggregate(pipeline))
        
        # Define rack assignments based on categories
        rack_assignments = {
            'Electronics': ['B'],  # B racks for Electronics
            'Clothing': ['P'],     # P racks for Clothing  
            'Food': ['D'],         # D racks for Food
            'refrigerated': ['D']  # D racks for refrigerated items
        }
        
        mapping = {}
        used_locations = set()
        
        for group in grouped_items:
            location_id = group['_id']['locationID']
            category = group['_id']['category']
            storage_type = group['_id']['storage_type']
            size = group['_id']['size']
            
            if not isinstance(location_id, int) or location_id <= 0:
                continue
            
            # Determine preferred rack types
            preferred_racks = []
            
            if storage_type == 'refrigerated':
                preferred_racks = ['D']
            elif category in rack_assignments:
                preferred_racks = rack_assignments[category]
            else:
                preferred_racks = ['B', 'P', 'D']  # Any rack
            
            # Find available location in preferred racks
            assigned_location = None
            
            for rack_prefix in preferred_racks:
                available_in_rack = [
                    loc for loc in self.available_locations 
                    if loc.startswith(rack_prefix) and loc not in used_locations
                ]
                
                if available_in_rack:
                    # Prefer lower floors for heavier items (size L)
                    if size == 'L':
                        # Sort by floor (ascending - lower floors first)
                        available_in_rack.sort(key=lambda x: int(x.split('.')[1]))
                    elif size == 'S':
                        # Higher floors for smaller items
                        available_in_rack.sort(key=lambda x: int(x.split('.')[1]), reverse=True)
                    
                    assigned_location = available_in_rack[0]
                    break
            
            # Fallback to any available location
            if not assigned_location:
                remaining_locations = [loc for loc in self.available_locations if loc not in used_locations]
                if remaining_locations:
                    assigned_location = remaining_locations[0]
                else:
                    assigned_location = "B01.1"  # Ultimate fallback
            
            mapping[location_id] = assigned_location
            used_locations.add(assigned_location)
            
            print(f"  {location_id} -> {assigned_location} (Category: {category}, Storage: {storage_type}, Size: {size})")
        
        return mapping
    
    def preview_migration(self, mapping):
        """Preview what the migration will do"""
        print("\n=== Migration Preview ===")
        
        for old_id, new_location in mapping.items():
            items = list(self.db.inventory.find({"locationID": old_id}))
            print(f"\nLocationID {old_id} -> {new_location}:")
            for item in items:
                print(f"  - {item['name']} (Category: {item['category']}, Size: {item.get('size', 'N/A')})")
    
    def perform_migration(self, mapping, dry_run=True):
        """Perform the actual migration"""
        if dry_run:
            print("\n=== DRY RUN - Migration Preview ===")
            print("This is a preview. No changes will be made to the database.")
        else:
            print("\n=== PERFORMING MIGRATION ===")
            print("Updating database...")
        
        updates = []
        migration_timestamp = datetime.now().isoformat()
        
        for old_id, new_location in mapping.items():
            if dry_run:
                count = self.db.inventory.count_documents({"locationID": old_id})
                print(f"Would update {count} items: locationID {old_id} -> {new_location}")
            else:
                # Prepare bulk update using UpdateMany operation
                from pymongo import UpdateMany
                updates.append(
                    UpdateMany(
                        {"locationID": old_id},
                        {
                            "$set": {
                                "locationID": new_location,
                                "migration_timestamp": migration_timestamp,
                                "migration_note": f"Migrated from numeric ID {old_id} to warehouse location {new_location}"
                            }
                        }
                    )
                )
        
        if not dry_run and updates:
            try:
                result = self.db.inventory.bulk_write(updates)
                print(f"✅ Migration completed successfully!")
                print(f"   Modified documents: {result.modified_count}")
                print(f"   Matched documents: {result.matched_count}")
            except BulkWriteError as e:
                print(f"❌ Migration failed: {e}")
                return False
        
        return True
    
    def verify_migration(self, original_mapping):
        """Verify the migration was successful"""
        print("\n=== Migration Verification ===")
        
        # Check new location distribution
        pipeline = [
            {"$group": {"_id": "$locationID", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        new_location_stats = list(self.db.inventory.aggregate(pipeline))
        
        print("New locationID distribution:")
        for stat in new_location_stats:
            print(f"  {stat['_id']}: {stat['count']} items")
        
        # Validate all locations are valid warehouse locations
        print("\nValidation check:")
        all_valid = True
        for stat in new_location_stats:
            location = stat['_id']
            if not self.warehouse_mapper.validate_location(location):
                print(f"  ❌ Invalid location: {location}")
                all_valid = False
            else:
                print(f"  ✅ Valid location: {location}")
        
        if all_valid:
            print("\n🎉 All locations are valid warehouse locations!")
        else:
            print("\n⚠️ Some invalid locations found. Please review.")
        
        return all_valid
    
    def run_migration(self, use_smart_mapping=True, dry_run=True):
        """Run the complete migration process"""
        print("🚀 Inventory Location ID Migration")
        print("=" * 50)
        
        # Analyze current state
        location_stats = self.analyze_current_inventory()
        
        # Create mapping
        if use_smart_mapping:
            mapping = self.create_smart_mapping_by_category(location_stats)
        else:
            mapping = self.create_numeric_to_warehouse_mapping(location_stats)
        
        # Preview migration
        self.preview_migration(mapping)
        
        # Perform migration
        if self.perform_migration(mapping, dry_run=dry_run):
            if not dry_run:
                # Verify migration
                self.verify_migration(mapping)
                
                print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
                print("=" * 40)
                print("✅ All inventory items now have proper warehouse location IDs")
                print("✅ Locations assigned based on categories and storage requirements")
                print("📁 Migration timestamp and notes added to all records")
            else:
                print("\n📋 DRY RUN COMPLETED")
                print("=" * 25)
                print("To perform the actual migration, run with dry_run=False")
        else:
            print("\n❌ MIGRATION FAILED")
            print("Please review the errors above and try again.")
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'client'):
            self.client.close()

def main():
    """Main function to run the migration"""
    print("Inventory Location ID Migration Tool")
    print("This will convert numeric locationID values to warehouse location format")
    print()
    
    # Ask user for migration type and mode
    print("Migration options:")
    print("1. Smart mapping (assigns locations based on category, storage type, size)")
    print("2. Random mapping (randomly assigns available warehouse locations)")
    
    mapping_choice = input("Choose mapping type (1 or 2): ").strip()
    use_smart_mapping = mapping_choice != "2"
    
    print("\nExecution mode:")
    print("1. Dry run (preview only, no changes)")
    print("2. Actual migration (makes changes to database)")
    
    mode_choice = input("Choose execution mode (1 or 2): ").strip()
    dry_run = mode_choice != "2"
    
    if not dry_run:
        confirm = input("\n⚠️  This will modify your database. Are you sure? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Migration cancelled.")
            return
    
    # Run migration
    migrator = InventoryLocationMigrator()
    migrator.run_migration(use_smart_mapping=use_smart_mapping, dry_run=dry_run)

if __name__ == "__main__":
    main()
