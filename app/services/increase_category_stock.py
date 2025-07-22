from app.utils.database import get_collection

def increase_category_stock_by_category(category_values):
    category_stock_col = get_collection("category_stock")
    print("Connected to collection:", category_stock_col.full_name)
    print("Existing categories in DB:")
    for doc in category_stock_col.find({}, {"category": 1, "_id": 0}):
        print(f"'{doc.get('category')}'")
    total_updated = 0
    for category, new_value in category_values.items():
        print(f"\nUpdating category '{category}':")
        before = category_stock_col.find_one({"category": category})
        print("  Before:", before)
        result = category_stock_col.update_one(
            {"category": category},
            {
                "$set": {
                    "stock": new_value,
                    "stock_level": new_value,
                    "total_stock": new_value
                }
            }
        )
        after = category_stock_col.find_one({"category": category})
        print(f"  Matched: {result.matched_count}, Modified: {result.modified_count}")
        print("  After:", after)
        total_updated += result.modified_count
    print(f"Total updated: {total_updated} documents.")

if __name__ == "__main__":
    # Example: set different values for each category
    category_values = {
        "electronics": 502200,
        "clothing": 800450,
        "books_media": 350300,
        "home_garden": 761000,
        "sports_outdoors": 68400,
        "health_beauty": 505000
    }
    increase_category_stock_by_category(category_values)

    # Manual update and before/after print for debugging
    category_stock_col = get_collection("category_stock")
    print("\nManual update debug for 'electronics':")
    print("Before update:", category_stock_col.find_one({"category": "electronics"}))
    result = category_stock_col.update_one(
        {"category": "electronics"},
        {"$set": {"stock": 90999, "stock_level": 90999, "total_stock": 90999}}
    )
    print("Manual update result:", result.raw_result)
    print("After update:", category_stock_col.find_one({"category": "electronics"}))
