# create_seasonal_demand.py
# Run this script to create seasonal_demand collection with sample data

import pymongo
from datetime import datetime
import random


MONGO_URI = "mongodb+srv://judithfdo2002:kTCN07mlhHmtgrt0@cluster0.9wwflqj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0" 
DATABASE_NAME = "warehouse_management"     

def create_seasonal_demand_collection():
    """Create seasonal_demand collection with sample data"""
    
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        
        # Get existing items from inventory collection
        inventory_collection = db["inventory"]
        existing_items = list(inventory_collection.find({}, {"itemID": 1, "category": 1}))
        
        if not existing_items:
            print("No items found in inventory collection. Creating sample items...")
            # Create some sample items if inventory is empty
            sample_items = [
                {"itemID": 1, "category": "Electronics"},
                {"itemID": 2, "category": "Clothing"},
                {"itemID": 3, "category": "Food"},
                {"itemID": 4, "category": "Books"},
                {"itemID": 5, "category": "Toys"}
            ]
            existing_items = sample_items
        
        # Create seasonal demand data
        seasonal_demand_data = []
        seasons = ["Winter", "Spring", "Summer", "Fall"]
        
        for item in existing_items:
            item_id = item["itemID"]
            category = item.get("category", "General")
            
            for season in seasons:
                # Generate seasonal score based on category and season
                seasonal_score = generate_seasonal_score(category, season)
                
                seasonal_demand_data.append({
                    "itemID": item_id,
                    "season": season,
                    "seasonal_score": seasonal_score,
                    "category": category,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                })
        
        # Insert data into seasonal_demand collection
        seasonal_collection = db["seasonal_demand"]
        
        # Clear existing data (optional - remove this if you want to keep existing data)
        seasonal_collection.delete_many({})
        
        # Insert new data
        result = seasonal_collection.insert_many(seasonal_demand_data)
        
        print(f"✅ Successfully created seasonal_demand collection")
        print(f"📊 Inserted {len(result.inserted_ids)} seasonal demand records")
        print(f"📋 Covered {len(existing_items)} items across {len(seasons)} seasons")
        
        # Create index for better performance
        seasonal_collection.create_index([("itemID", 1), ("season", 1)])
        print("📈 Created index on itemID and season")
        
        # Show sample data
        print("\n📄 Sample records:")
        samples = list(seasonal_collection.find().limit(5))
        for sample in samples:
            print(f"  Item {sample['itemID']} - {sample['season']}: {sample['seasonal_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Error creating seasonal_demand collection: {str(e)}")
    
    finally:
        client.close()

def generate_seasonal_score(category, season):
    """Generate realistic seasonal scores based on your actual categories and seasons"""
    
    # Updated seasonal patterns for your actual categories
    seasonal_patterns = {
        "clothing": {
            "Winter": 0.9,  # High demand (winter clothes, coats, boots)
            "Spring": 0.7,  # Spring fashion, lighter clothes
            "Summer": 0.6,  # Summer clothes but lower overall
            "Fall": 0.8     # Fall fashion, back-to-school clothes
        },
        "books_media": {
            "Winter": 0.7,  # Holiday reading, indoor activities
            "Spring": 0.6,  # Moderate demand
            "Summer": 0.5,  # Lower demand (people outdoors)
            "Fall": 0.8     # Back to school, new releases
        },
        "health_beauty": {
            "Winter": 0.8,  # Skincare for dry weather, self-care
            "Spring": 0.7,  # Spring cleaning, fresh start
            "Summer": 0.9,  # Sunscreen, summer beauty products
            "Fall": 0.6     # Moderate demand
        },
        "electronics": {
            "Winter": 0.9,  # Holiday shopping, Black Friday
            "Spring": 0.6,  # Moderate demand
            "Summer": 0.5,  # Lower demand 
            "Fall": 0.7     # Back to school electronics
        },
        "home_garden": {
            "Winter": 0.5,  # Lower gardening activity
            "Spring": 0.9,  # Peak gardening season
            "Summer": 0.8,  # High gardening activity
            "Fall": 0.7     # Fall planting, preparation
        },
        "sports_outdoor": {
            "Winter": 0.6,  # Winter sports equipment
            "Spring": 0.8,  # Spring sports, hiking gear
            "Summer": 0.9,  # Peak outdoor season
            "Fall": 0.7     # Fall sports, hunting season
        },
        "toys": {
            "Winter": 0.9,  # Holiday shopping
            "Spring": 0.5,  # Lower demand
            "Summer": 0.6,  # Outdoor toys, vacation
            "Fall": 0.7     # Back to school, new releases
        },
        "automotive": {
            "Winter": 0.8,  # Winter tires, antifreeze, accessories
            "Spring": 0.7,  # Spring maintenance
            "Summer": 0.6,  # Travel accessories
            "Fall": 0.7     # Winter preparation
        },
        "food_grocery": {
            "Winter": 0.7,  # Holiday cooking, comfort foods
            "Spring": 0.6,  # Fresh ingredients
            "Summer": 0.8,  # BBQ, fresh produce, beverages
            "Fall": 0.7     # Back to school lunches, harvest
        },
        "pet_supplies": {
            "Winter": 0.6,  # Indoor pet activities
            "Spring": 0.7,  # Spring cleaning, new accessories
            "Summer": 0.8,  # Outdoor pet activities
            "Fall": 0.6     # Moderate demand
        },
        "office_supplies": {
            "Winter": 0.6,  # Holiday cards, year-end supplies
            "Spring": 0.7,  # Spring organization
            "Summer": 0.5,  # Lower demand
            "Fall": 0.9     # Back to school/work peak
        },
        "baby_kids": {
            "Winter": 0.8,  # Holiday gifts, winter clothes
            "Spring": 0.7,  # Spring clothes, outdoor toys
            "Summer": 0.8,  # Summer activities, vacation items
            "Fall": 0.8     # Back to school supplies
        },
        "jewelry_accessories": {
            "Winter": 0.9,  # Holiday gifts, formal events
            "Spring": 0.6,  # Moderate demand
            "Summer": 0.7,  # Summer accessories
            "Fall": 0.6     # Moderate demand
        },
        "tools_hardware": {
            "Winter": 0.6,  # Indoor projects
            "Spring": 0.9,  # Peak home improvement season
            "Summer": 0.8,  # Outdoor projects
            "Fall": 0.7     # Winter preparation projects
        },
        "general": {
            "Winter": 0.6,
            "Spring": 0.6,
            "Summer": 0.6,
            "Fall": 0.6
        }
    }
    
    # Normalize category name (handle variations)
    normalized_category = category.lower().replace(" ", "_").replace("-", "_")
    
    # Get base score for category and season
    if normalized_category in seasonal_patterns:
        base_score = seasonal_patterns[normalized_category].get(season, 0.5)
    else:
        # If category not found, use general pattern
        base_score = seasonal_patterns["general"].get(season, 0.5)
        print(f"⚠️  Unknown category '{category}', using general pattern")
    
    # Add some randomness (±0.1) to make it more realistic
    import random
    random_factor = random.uniform(-0.1, 0.1)
    final_score = max(0.1, min(1.0, base_score + random_factor))
    
    return round(final_score, 2)