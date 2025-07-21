#!/usr/bin/env python3
"""
Run Inventory Location Migration - Direct Execution
"""

import os
import sys

# Import the migrator
sys.path.append(os.path.dirname(__file__))
from migrate_inventory_location_ids import InventoryLocationMigrator

def run_migration_directly():
    """Run the migration with predefined settings"""
    print("🚀 Running Inventory Location ID Migration")
    print("Settings: Smart mapping enabled, Actual migration")
    print("=" * 50)
    
    migrator = InventoryLocationMigrator()
    migrator.run_migration(use_smart_mapping=True, dry_run=False)

if __name__ == "__main__":
    run_migration_directly()
