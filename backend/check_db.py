"""
Script to check what's actually in the database
"""
from pymongo import MongoClient

# Try both databases
for db_name in ["lmsfull", "safety_ai"]:
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client[db_name]
        
        print(f"\n{'='*50}")
        print(f"Database: {db_name}")
        print(f"{'='*50}")
        
        collections = db.list_collection_names()
        print(f"Collections: {collections}")
        
        for collection_name in ["employees", "users", "workers"]:
            if collection_name in collections:
                count = db[collection_name].count_documents({})
                print(f"{collection_name}: {count} documents")
                
                # Get one sample document
                if count > 0:
                    sample = db[collection_name].find_one()
                    print(f"  Sample: {sample}")
            else:
                print(f"{collection_name}: does not exist")
        
        client.close()
    except Exception as e:
        print(f"Error checking {db_name}: {e}")
