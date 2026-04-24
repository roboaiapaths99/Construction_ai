"""
Unified Database Service for AI Construction System
Uses MongoDB for all operations (Workers, Attendance, Violations, Alerts)
"""
from typing import Optional, List, Dict, Tuple
import numpy as np
import json
from datetime import datetime, timedelta
from bson import ObjectId
from .mongodb import MongoDBConfig

class DatabaseService:
    def __init__(self):
        self.db = MongoDBConfig.get_database()
        if self.db is None:
            print("❌ Critical: Could not connect to MongoDB")

    # =========================================================
    # WORKER OPERATIONS
    # =========================================================
    def add_worker(self, worker_id: str, name: str, email: str = None, phone: str = None) -> bool:
        """Add a new worker"""
        try:
            worker_data = {
                "worker_id": worker_id,
                "name": name,
                "email": email,
                "phone": phone,
                "status": "active",
                "created_at": datetime.utcnow()
            }
            self.db.workers.insert_one(worker_data)
            return True
        except Exception as e:
            print(f"Error adding worker: {e}")
            return False

    def get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get a worker by ID"""
        try:
            worker = self.db.workers.find_one({"worker_id": worker_id})
            if worker:
                worker["_id"] = str(worker["_id"])
            return worker
        except Exception as e:
            print(f"Error getting worker: {e}")
            return None

    def worker_exists(self, worker_id: str) -> bool:
        """Check if worker exists"""
        return self.db.workers.find_one({"worker_id": worker_id}) is not None

    def get_all_workers(self) -> List[Dict]:
        """Get all active workers"""
        try:
            workers = list(self.db.workers.find({"status": "active"}))
            for worker in workers:
                worker["_id"] = str(worker["_id"])
            return workers
        except Exception as e:
            print(f"Error getting all workers: {e}")
            return []

    def delete_worker(self, worker_id: str) -> bool:
        """Delete worker and related data"""
        try:
            # Get employee object ID if it exists
            # (In some collections we might use worker_id as string, in others as ObjectId)
            self.db.attendance_logs.delete_many({"worker_id": worker_id})
            self.db.employee_face_profiles.delete_many({"worker_id": worker_id})
            self.db.workers.delete_one({"worker_id": worker_id})
            return True
        except Exception as e:
            print(f"Error deleting worker: {e}")
            return False

    # =========================================================
    # FACE EMBEDDING OPERATIONS
    # =========================================================
    def store_embedding(self, worker_id: str, embedding: np.ndarray) -> bool:
        """Store face embedding"""
        try:
            embedding_list = embedding.tolist()
            profile_data = {
                "worker_id": worker_id,
                "embedding": embedding_list,
                "created_at": datetime.utcnow()
            }
            self.db.employee_face_profiles.update_one(
                {"worker_id": worker_id},
                {"$set": profile_data},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all worker embeddings for cache"""
        try:
            profiles = list(self.db.employee_face_profiles.find({}))
            embeddings = {}
            for profile in profiles:
                embeddings[profile["worker_id"]] = np.array(profile["embedding"])
            return embeddings
        except Exception as e:
            print(f"Error getting all embeddings: {e}")
            return {}

    # =========================================================
    # ATTENDANCE OPERATIONS
    # =========================================================
    def mark_attendance(self, worker_id: str, event_type: str = "check_in", confidence: float = 1.0, detected_by: str = "webcam") -> bool:
        """Mark attendance in MongoDB"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            if event_type == "check_in":
                # Check for existing open record
                existing = self.db.attendance_logs.find_one({
                    "worker_id": worker_id,
                    "date": today,
                    "check_out_time": None
                })
                
                if existing:
                    return True
                
                log_data = {
                    "worker_id": worker_id,
                    "date": today,
                    "check_in_time": datetime.utcnow(),
                    "check_out_time": None,
                    "status": "incomplete",
                    "marked_by": detected_by,
                    "confidence": confidence
                }
                self.db.attendance_logs.insert_one(log_data)
                return True
                
            elif event_type == "check_out":
                result = self.db.attendance_logs.update_one(
                    {
                        "worker_id": worker_id,
                        "date": today,
                        "check_out_time": None
                    },
                    {
                        "$set": {
                            "check_out_time": datetime.utcnow(),
                            "status": "present"
                        }
                    }
                )
                return result.modified_count > 0
                
            return False
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False

    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance logs with worker names"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            # Using aggregation to join with workers
            pipeline = [
                {"$match": {"date": today}},
                {
                    "$lookup": {
                        "from": "workers",
                        "localField": "worker_id",
                        "foreignField": "worker_id",
                        "as": "worker_info"
                    }
                },
                {"$unwind": "$worker_info"},
                {
                    "$project": {
                        "_id": 0,
                        "worker_id": 1,
                        "name": "$worker_info.name",
                        "check_in": "$check_in_time",
                        "check_out": "$check_out_time",
                        "detected_by": "$marked_by",
                        "confidence": 1
                    }
                }
            ]
            records = list(self.db.attendance_logs.aggregate(pipeline))
            return records
        except Exception as e:
            print(f"Error getting today's attendance: {e}")
            return []

    # =========================================================
    # DASHBOARD & ANALYTICS
    # =========================================================
    def get_dashboard_stats(self) -> Dict:
        """Calculate real-time dashboard stats"""
        try:
            total_violations = self.db.violations.count_documents({})
            total_alerts = self.db.alerts.count_documents({"status": "active"})
            active_workers = self.db.workers.count_documents({"status": "active"})
            
            # Last 5 violations
            latest_violations = list(self.db.violations.find().sort("timestamp", -1).limit(5))
            for v in latest_violations:
                v["_id"] = str(v["_id"])
            
            return {
                "total_violations": total_violations,
                "total_alerts": total_alerts,
                "active_workers": active_workers,
                "latest_violations": latest_violations,
                "uptime_seconds": int(datetime.utcnow().timestamp())
            }
        except Exception as e:
            print(f"Error getting dashboard stats: {e}")
            return {}

# Global instance
db_service = DatabaseService()
