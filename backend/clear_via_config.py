"""
Script to clear employees using the same MongoDBConfig as the server
"""
import sys
sys.path.insert(0, '.')

from config.mongodb import MongoDBConfig

try:
    db = MongoDBConfig.get_database()
    if db is None:
        print("Failed to connect to database")
        exit(1)
    
    print(f"Connected to database: {db.name}")
    
    # Check employees count
    employees_count = db.employees.count_documents({})
    print(f"Employees count before delete: {employees_count}")
    
    if employees_count > 0:
        result = db.employees.delete_many({})
        print(f"Deleted {result.deleted_count} employees")
    else:
        print("No employees to delete")
    
    # Verify
    employees_count_after = db.employees.count_documents({})
    print(f"Employees count after delete: {employees_count_after}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
