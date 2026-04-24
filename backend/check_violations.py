"""
Script to check if AI detection is working by checking violations in database
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
    
    # Check violations collection
    violations_count = db.violations.count_documents({})
    print(f"Total violations in database: {violations_count}")
    
    # Get recent violations
    if violations_count > 0:
        recent_violations = list(db.violations.find().sort("created_at", -1).limit(5))
        print(f"\nRecent violations:")
        for v in recent_violations:
            print(f"  - {v.get('type', 'Unknown')} at {v.get('created_at', 'Unknown')}")
    else:
        print("No violations found in database")
    
    # Check alerts collection
    alerts_count = db.alerts.count_documents({})
    print(f"\nTotal alerts in database: {alerts_count}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
