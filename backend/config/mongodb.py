"""
MongoDB configuration for AI Construction System
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class MongoDBConfig:
    """MongoDB configuration class"""
    
    # MongoDB connection settings
    # Support both local MongoDB and MongoDB Atlas
    MONGODB_URL = os.getenv("MONGODB_URL", None)
    HOST = os.getenv("MONGO_HOST", "localhost")
    PORT = int(os.getenv("MONGO_PORT", 27017))
    USERNAME = os.getenv("MONGO_USERNAME", "")
    PASSWORD = os.getenv("MONGO_PASSWORD", "")
    DATABASE = os.getenv("MONGO_DB_NAME", os.getenv("DATABASE_NAME", "safety_ai"))
    
    # Connection settings
    TIMEOUT = int(os.getenv("MONGO_TIMEOUT", 5000))
    
    _client = None
    _db = None
    
    @staticmethod
    def get_connection_string():
        """Get MongoDB connection string"""
        # If MONGODB_URL is provided (MongoDB Atlas), use it directly
        if MongoDBConfig.MONGODB_URL:
            return MongoDBConfig.MONGODB_URL
        
        # Otherwise, build connection string for local MongoDB
        if MongoDBConfig.USERNAME and MongoDBConfig.PASSWORD:
            return f"mongodb://{MongoDBConfig.USERNAME}:{MongoDBConfig.PASSWORD}@{MongoDBConfig.HOST}:{MongoDBConfig.PORT}/{MongoDBConfig.DATABASE}"
        else:
            return f"mongodb://{MongoDBConfig.HOST}:{MongoDBConfig.PORT}/{MongoDBConfig.DATABASE}"
    
    @staticmethod
    def get_client():
        """Get MongoDB client"""
        try:
            if MongoDBConfig._client is None:
                connection_string = MongoDBConfig.get_connection_string()
                # Added more robust connection options
                MongoDBConfig._client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=MongoDBConfig.TIMEOUT,
                    connectTimeoutMS=MongoDBConfig.TIMEOUT,
                    retryWrites=True,
                    # Fallback for older TLS environments if needed
                    tlsAllowInvalidCertificates=False 
                )
            return MongoDBConfig._client
        except Exception as e:
            print(f"❌ MongoDB client initialization error: {e}")
            return None
    
    @staticmethod
    def get_database():
        """Get MongoDB database"""
        try:
            client = MongoDBConfig.get_client()
            if client is None:
                return None
            
            # Use DATABASE_NAME or fallback to safety_ai
            db_name = MongoDBConfig.DATABASE
            db = client[db_name]
            
            # Trigger an actual connection check
            client.admin.command('ping')
            
            if MongoDBConfig._db is None:
                print(f"✅ Successfully connected to MongoDB database: {db_name}")
                MongoDBConfig._db = db
                
            return db
        except Exception as e:
            print(f"❌ Database connection error (handshake failed?): {e}")
            if "SSL handshake failed" in str(e):
                print("💡 TIP: Check if your VPS IP is whitelisted in MongoDB Atlas Network Access!")
            return None
    
    @staticmethod
    def test_connection():
        """Test MongoDB connection"""
        try:
            client = MongoDBConfig.get_client()
            if client:
                # The ping command is cheap and does not require auth
                client.admin.command('ping')
                print("✅ MongoDB connection test passed")
                return True
            return False
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"❌ MongoDB test failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during MongoDB test: {e}")
            return False

def initialize_mongodb():
    """Initialize MongoDB with collections and indexes"""
    try:
        db = MongoDBConfig.get_database()
        if db is None:
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Create collections with validation schemas
        collections_config = {
            "workers": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["name"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "name": {"bsonType": "string"},
                            "role": {"bsonType": "string"},
                            "status": {
                                "enum": ["Active", "Inactive", "At Risk"],
                                "description": "Worker status"
                            },
                            "location": {"bsonType": "string"},
                            "created_at": {"bsonType": "date"},
                            "updated_at": {"bsonType": "date"}
                        }
                    }
                }
            },
            "violations": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["camera_name"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "worker_id": {"bsonType": "objectId"},
                            "camera_name": {"bsonType": "string"},
                            "violation_type": {
                                "enum": ["No Helmet", "No Vest", "Multiple Violations", "Unauthorized"],
                                "description": "Type of violation"
                            },
                            "confidence": {"bsonType": "double"},
                            "bbox_x": {"bsonType": "int"},
                            "bbox_y": {"bsonType": "int"},
                            "bbox_width": {"bsonType": "int"},
                            "bbox_height": {"bsonType": "int"},
                            "image_path": {"bsonType": "string"},
                            "status": {
                                "enum": ["open", "resolved", "ignored"],
                                "description": "Violation status"
                            },
                            "timestamp": {"bsonType": "date"}
                        }
                    }
                }
            },
            "alerts": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["message"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "message": {"bsonType": "string"},
                            "level": {
                                "enum": ["low", "medium", "high"],
                                "description": "Alert severity level"
                            },
                            "violation_id": {"bsonType": "objectId"},
                            "camera_name": {"bsonType": "string"},
                            "status": {
                                "enum": ["active", "acknowledged", "resolved"],
                                "description": "Alert status"
                            },
                            "created_at": {"bsonType": "date"},
                            "resolved_at": {"bsonType": "date"}
                        }
                    }
                }
            },
            "cameras": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["name"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "name": {"bsonType": "string"},
                            "ip": {"bsonType": "string"},
                            "location": {"bsonType": "string"},
                            "status": {
                                "enum": ["active", "inactive", "maintenance"],
                                "description": "Camera status"
                            },
                            "type": {
                                "enum": ["rtsp", "webcam", "ip_camera"],
                                "description": "Camera type"
                            },
                            "rtsp_url": {"bsonType": "string"},
                            "username": {"bsonType": "string"},
                            "password": {"bsonType": "string"},
                            "port": {"bsonType": "int"},
                            "created_at": {"bsonType": "date"}
                        }
                    }
                }
            },
            "incidents": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["camera_name"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "camera_name": {"bsonType": "string"},
                            "violation_type": {"bsonType": "string"},
                            "confidence": {"bsonType": "double"},
                            "bbox_x": {"bsonType": "int"},
                            "bbox_y": {"bsonType": "int"},
                            "bbox_width": {"bsonType": "int"},
                            "bbox_height": {"bsonType": "int"},
                            "timestamp": {"bsonType": "date"},
                            "image_path": {"bsonType": "string"}
                        }
                    }
                }
            },
            "employees": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["name"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "name": {"bsonType": "string"},
                            "email": {"bsonType": "string"},
                            "phone": {"bsonType": "string"},
                            "employee_code": {"bsonType": "string"},
                            "status": {"bsonType": "string"},
                            "created_at": {"bsonType": "date"}
                        }
                    }
                }
            },
            "employee_face_profiles": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["employee_id", "embedding"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "employee_id": {"bsonType": "objectId"},
                            "photo_url": {"bsonType": "string"},
                            "cropped_face_url": {"bsonType": "string"},
                            "embedding": {"bsonType": "array"},
                            "quality_score": {"bsonType": "double"},
                            "is_primary": {"bsonType": "bool"},
                            "created_at": {"bsonType": "date"}
                        }
                    }
                }
            },
            "attendance_logs": {
                "validator": {
                    "$jsonSchema": {
                        "bsonType": "object",
                        "required": ["employee_id", "date"],
                        "properties": {
                            "_id": {"bsonType": "objectId"},
                            "employee_id": {"bsonType": "objectId"},
                            "camera_id": {"bsonType": "objectId"},
                            "date": {"bsonType": "string"},
                            "check_in_time": {"bsonType": "date"},
                            "check_out_time": {"bsonType": "date"},
                            "source": {"bsonType": "string"},
                            "confidence": {"bsonType": "double"},
                            "status": {"bsonType": "string"},
                            "marked_by": {"bsonType": "string"},
                            "worker_id": {"bsonType": "string"}
                        }
                    }
                }
            }
        }
        
        for collection_name, config in collections_config.items():
            try:
                if collection_name in db.list_collection_names():
                    print(f"✅ Collection '{collection_name}' already exists")
                else:
                    db.create_collection(collection_name, **config)
                    print(f"✅ Collection '{collection_name}' created successfully")
                
                # Create indexes
                if collection_name == "violations":
                    db[collection_name].create_index("worker_id")
                    db[collection_name].create_index("camera_name")
                    db[collection_name].create_index("timestamp")
                
                elif collection_name == "alerts":
                    db[collection_name].create_index("violation_id")
                    db[collection_name].create_index("camera_name")
                    db[collection_name].create_index("created_at")
                
                elif collection_name == "incidents":
                    db[collection_name].create_index("camera_name")
                    db[collection_name].create_index("timestamp")
                
                elif collection_name == "workers":
                    db[collection_name].create_index("name")
                
                elif collection_name == "cameras":
                    db[collection_name].create_index("name")
                    db[collection_name].create_index("ip")
                
            except Exception as e:
                print(f"⚠️ Error creating collection '{collection_name}': {e}")
        
        print("✅ Database initialization completed")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def get_database():
    """Get the MongoDB database instance"""
    return MongoDBConfig.get_database()
