"""
Script to clear all employee enrollments and face profiles
"""
from pymongo import MongoClient
import os

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "lmsfull"

try:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    print(f"Connected to MongoDB: {DB_NAME}")
    
    # List all collections
    collections = db.list_collection_names()
    print(f"Collections: {collections}")
    
    # Clear ALL collections that might contain employee data
    collections_to_clear = [
        'employees',
        'employee_face_profiles',
        'workers',
        'users',  # Add users collection
        'attendance_logs',
        'training_records',
        'exit_management',
        'payrolls',
        'onboardings',
        'leave_requests',
        'performance_reviews'
    ]
    
    total_deleted = 0
    for collection_name in collections_to_clear:
        if collection_name in collections:
            # Delete without checking count first
            result = db[collection_name].delete_many({})
            if result.deleted_count > 0:
                total_deleted += result.deleted_count
                print(f"✅ Deleted {result.deleted_count} documents from {collection_name}")
            else:
                print(f"ℹ️ No documents in {collection_name}")
        else:
            print(f"ℹ️ Collection {collection_name} does not exist")
    
    print(f"\n✅ Total documents deleted: {total_deleted}")
    print(f"✅ All employee-related data cleared successfully")
    
    client.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
