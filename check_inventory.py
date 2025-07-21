from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["warehouse_management"]

print("=== Checking Inventory ===")
items = list(db.inventory.find({}))
print(f"Total items in inventory: {len(items)}")

print("\nFirst 5 items:")
for i, item in enumerate(items[:5]):
    print(f"{i+1}. ID: {item.get('itemID')}, Name: {item.get('itemName')}, Qty: {item.get('quantity', 0)}")

print("\nItems with quantity > 0:")
items_with_stock = list(db.inventory.find({"quantity": {"$gt": 0}}))
print(f"Items with stock: {len(items_with_stock)}")

for item in items_with_stock[:5]:
    print(f"  - ID: {item.get('itemID')}, Name: {item.get('itemName')}, Qty: {item.get('quantity')}")

client.close()
