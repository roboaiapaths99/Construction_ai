"""
Script to directly clear employees from MongoDB using pymongo
"""
from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017")
    db = client["lmsfull"]
    
    print(f"Connected to database: {db.name}")
    
    # Check employees collection
    employees_count = db.employees.count_documents({})
    print(f"Employees count before delete: {employees_count}")
    
    # Delete all employees
    result = db.employees.delete_many({})
    print(f"Deleted {result.deleted_count} employees")
    
    # Verify deletion
    employees_count_after = db.employees.count_documents({})
    print(f"Employees count after delete: {employees_count_after}")
    
    client.close()
    
except Exception as e:
    print(f"Error: {e}")
