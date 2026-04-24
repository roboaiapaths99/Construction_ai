"""
Worker and attendance database management
Handles worker profiles, face embeddings, and attendance records
"""
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/workers.db")

class WorkerDatabase:
    """SQLite database for workers and attendance tracking"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Workers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    enrollment_date TEXT DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    face_embedding TEXT
                )
            """)
            
            # Attendance records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT NOT NULL,
                    check_in_time TEXT,
                    check_out_time TEXT,
                    attendance_date TEXT,
                    detected_by TEXT DEFAULT 'manual',
                    confidence REAL DEFAULT 0.0,
                    FOREIGN KEY(worker_id) REFERENCES workers(worker_id)
                )
            """)
            
            # Face embeddings cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    worker_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(worker_id) REFERENCES workers(worker_id)
                )
            """)
            
            conn.commit()
            logger.info("✅ Worker database initialized")
            conn.close()
        except Exception as e:
            logger.error(f"❌ Database init error: {e}")
    
    def add_worker(self, worker_id: str, name: str, email: str = None, phone: str = None) -> bool:
        """Add a new worker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workers (worker_id, name, email, phone)
                VALUES (?, ?, ?, ?)
            """, (worker_id, name, email, phone))
            conn.commit()
            conn.close()
            logger.info(f"✅ Worker {worker_id} added")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding worker: {e}")
            return False
    
    def get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get a worker by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT worker_id, name, email, phone, enrollment_date, status
                FROM workers
                WHERE worker_id = ?
            """, (worker_id,))
            result = cursor.fetchone()
            conn.close()

            if not result:
                return None

            worker_id, name, email, phone, enrollment_date, status = result
            return {
                "worker_id": worker_id,
                "name": name,
                "email": email,
                "phone": phone,
                "enrollment_date": enrollment_date,
                "status": status
            }
        except Exception as e:
            logger.error(f"âŒ Error getting worker: {e}")
            return None

    def worker_exists(self, worker_id: str) -> bool:
        """Check whether a worker exists"""
        return self.get_worker(worker_id) is not None

    def store_embedding(self, worker_id: str, embedding: np.ndarray) -> bool:
        """Store face embedding for a worker"""
        try:
            embedding_json = json.dumps(embedding.tolist())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings_cache (worker_id, embedding, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (worker_id, embedding_json))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Embedding stored for worker {worker_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error storing embedding: {e}")
            return False
    
    def get_embedding(self, worker_id: str) -> Optional[np.ndarray]:
        """Retrieve face embedding for a worker"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings_cache WHERE worker_id = ?", (worker_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                embedding_json = result[0]
                embedding = np.array(json.loads(embedding_json))
                return embedding
            return None
        except Exception as e:
            logger.error(f"❌ Error getting embedding: {e}")
            return None
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all worker embeddings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT worker_id, embedding FROM embeddings_cache")
            results = cursor.fetchall()
            conn.close()
            
            embeddings = {}
            for worker_id, embedding_json in results:
                embeddings[worker_id] = np.array(json.loads(embedding_json))
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error getting embeddings: {e}")
            return {}
    
    def mark_attendance(self, worker_id: str, event_type: str = "check_in", confidence: float = 1.0, detected_by: str = "webcam") -> bool:
        """Mark attendance for a worker"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if event_type == "check_in":
                cursor.execute("""
                    SELECT id
                    FROM attendance
                    WHERE worker_id = ? AND attendance_date = ? AND check_out_time IS NULL
                    ORDER BY check_in_time DESC
                    LIMIT 1
                """, (worker_id, today))
                existing = cursor.fetchone()

                if existing:
                    conn.close()
                    logger.info(f"â„¹ï¸ Attendance already open for {worker_id} on {today}")
                    return True

                cursor.execute("""
                    INSERT INTO attendance (worker_id, check_in_time, attendance_date, detected_by, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (worker_id, datetime.now().isoformat(), today, detected_by, confidence))
            elif event_type == "check_out":
                cursor.execute("""
                    SELECT id
                    FROM attendance
                    WHERE worker_id = ? AND attendance_date = ? AND check_out_time IS NULL
                    ORDER BY check_in_time DESC
                    LIMIT 1
                """, (worker_id, today))
                existing = cursor.fetchone()

                if not existing:
                    conn.close()
                    logger.warning(f"âš ï¸ No open attendance record found for {worker_id} on {today}")
                    return False

                cursor.execute("""
                    UPDATE attendance 
                    SET check_out_time = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), existing[0]))
            else:
                conn.close()
                logger.warning(f"âš ï¸ Invalid attendance event type: {event_type}")
                return False
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Attendance marked for {worker_id}: {event_type}")
            return True
        except Exception as e:
            logger.error(f"❌ Error marking attendance: {e}")
            return False
    
    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.worker_id, w.name, a.check_in_time, a.check_out_time, a.detected_by, a.confidence
                FROM attendance a
                JOIN workers w ON a.worker_id = w.worker_id
                WHERE a.attendance_date = ?
                ORDER BY a.check_in_time DESC
            """, (today,))
            results = cursor.fetchall()
            conn.close()
            
            records = []
            for worker_id, name, check_in, check_out, detected_by, confidence in results:
                records.append({
                    "worker_id": worker_id,
                    "name": name,
                    "check_in": check_in,
                    "check_out": check_out,
                    "detected_by": detected_by,
                    "confidence": confidence
                })
            return records
        except Exception as e:
            logger.error(f"❌ Error getting attendance: {e}")
            return []
    
    def get_worker_attendance(self, worker_id: str, days: int = 30) -> List[Dict]:
        """Get attendance history for a worker"""
        try:
            date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT attendance_date, check_in_time, check_out_time
                FROM attendance
                WHERE worker_id = ? AND attendance_date >= ?
                ORDER BY attendance_date DESC
            """, (worker_id, date_from))
            results = cursor.fetchall()
            conn.close()
            
            records = []
            for date, check_in, check_out in results:
                records.append({
                    "date": date,
                    "check_in": check_in,
                    "check_out": check_out
                })
            return records
        except Exception as e:
            logger.error(f"❌ Error getting worker attendance: {e}")
            return []
    
    def delete_worker(self, worker_id: str) -> bool:
        """Delete a worker and all associated data (embeddings, attendance)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete attendance records
            cursor.execute("DELETE FROM attendance WHERE worker_id = ?", (worker_id,))
            
            # Delete embeddings
            cursor.execute("DELETE FROM embeddings_cache WHERE worker_id = ?", (worker_id,))
            
            # Delete worker
            cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Worker {worker_id} deleted with all associated data")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting worker: {e}")
            return False

    def get_all_workers(self) -> List[Dict]:
        """Get all active workers"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT worker_id, name, email, phone, enrollment_date, status
                FROM workers
                WHERE status = 'active'
                ORDER BY name
            """)
            results = cursor.fetchall()
            conn.close()
            
            workers = []
            for worker_id, name, email, phone, enrollment_date, status in results:
                workers.append({
                    "worker_id": worker_id,
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "enrollment_date": enrollment_date,
                    "status": status
                })
            return workers
        except Exception as e:
            logger.error(f"❌ Error getting workers: {e}")
            return []

# Global database instance
worker_db = WorkerDatabase()
