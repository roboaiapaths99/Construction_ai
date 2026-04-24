"""
Clear attendance and worker data for fresh testing
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE = os.getenv("MONGO_DB_NAME", os.getenv("DATABASE_NAME", "safety_ai"))

try:
    client = MongoClient(MONGODB_URL)
    db = client[DATABASE]
    
    print(f"Connected to database: {DATABASE}")
    
    # Clear attendance records
    attendance_count = db.attendance_records.count_documents({})
    if attendance_count > 0:
        db.attendance_records.delete_many({})
        print(f"✅ Cleared {attendance_count} attendance records")
    else:
        print("ℹ️ No attendance records to clear")
    
    # Clear worker face profiles
    face_profiles_count = db.employee_face_profiles.count_documents({})
    if face_profiles_count > 0:
        db.employee_face_profiles.delete_many({})
        print(f"✅ Cleared {face_profiles_count} worker face profiles")
    else:
        print("ℹ️ No worker face profiles to clear")
    
    # Clear workers (optional - uncomment if you want to remove all workers too)
    # workers_count = db.workers.count_documents({})
    # if workers_count > 0:
    #     db.workers.delete_many({})
    #     print(f"✅ Cleared {workers_count} workers")
    # else:
    #     print("ℹ️ No workers to clear")
    
    print("\n✅ Data cleared successfully for fresh testing")
    
except Exception as e:
    print(f"❌ Error clearing data: {e}")
