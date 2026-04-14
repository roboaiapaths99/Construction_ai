"""
Database configuration for AI Construction System
"""
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    """Database configuration class"""
    
    # Database connection settings
    HOST = os.getenv("DB_HOST", "localhost")
    USER = os.getenv("DB_USER", "root")
    PASSWORD = os.getenv("DB_PASSWORD", "root123")
    DATABASE = os.getenv("DB_NAME", "safety_ai")
    PORT = int(os.getenv("DB_PORT", 3306))
    
    # Connection pool settings
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", 5))
    POOL_NAME = os.getenv("DB_POOL_NAME", "safety_ai_pool")
    
    @staticmethod
    def get_connection_string():
        """Get database connection string"""
        return {
            "host": DatabaseConfig.HOST,
            "user": DatabaseConfig.USER,
            "password": DatabaseConfig.PASSWORD,
            "database": DatabaseConfig.DATABASE,
            "port": DatabaseConfig.PORT,
            "autocommit": True,
            "pool_size": DatabaseConfig.POOL_SIZE,
            "pool_name": DatabaseConfig.POOL_NAME
        }
    
    @staticmethod
    def get_connection():
        """Get database connection"""
        try:
            connection = mysql.connector.connect(**DatabaseConfig.get_connection_string())
            if connection.is_connected():
                print(f"✅ Connected to MySQL database: {DatabaseConfig.DATABASE}")
                return connection
            else:
                print("❌ Failed to connect to database")
                return None
        except Error as e:
            print(f"❌ Database connection error: {e}")
            return None
    
    @staticmethod
    def test_connection():
        """Test database connection"""
        try:
            connection = DatabaseConfig.get_connection()
            if connection and connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                connection.close()
                return result[0] == 1
            return False
        except Error as e:
            print(f"❌ Database test failed: {e}")
            return False

# Database table schemas
TABLE_SCHEMAS = {
    "workers": """
        CREATE TABLE IF NOT EXISTS workers (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            role VARCHAR(100),
            status ENUM('Active', 'Inactive', 'At Risk') DEFAULT 'Active',
            location VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """,
    
    "violations": """
        CREATE TABLE IF NOT EXISTS violations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            worker_id INT,
            camera_name VARCHAR(255),
            violation_type ENUM('No Helmet', 'No Vest', 'Multiple Violations', 'Unauthorized'),
            confidence FLOAT,
            bbox_x INT,
            bbox_y INT,
            bbox_width INT,
            bbox_height INT,
            image_path VARCHAR(500),
            status ENUM('open', 'resolved', 'ignored') DEFAULT 'open',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (worker_id) REFERENCES workers(id)
        )
    """,
    
    "alerts": """
        CREATE TABLE IF NOT EXISTS alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            message TEXT NOT NULL,
            level ENUM('low', 'medium', 'high') DEFAULT 'medium',
            violation_id INT,
            camera_name VARCHAR(255),
            status ENUM('active', 'acknowledged', 'resolved') DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP NULL,
            FOREIGN KEY (violation_id) REFERENCES violations(id)
        )
    """,
    
    "cameras": """
        CREATE TABLE IF NOT EXISTS cameras (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            ip VARCHAR(45),
            location VARCHAR(255),
            status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
            type ENUM('rtsp', 'webcam', 'ip_camera') DEFAULT 'webcam',
            rtsp_url TEXT,
            username VARCHAR(255),
            password VARCHAR(255),
            port INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
}

def initialize_database():
    """Initialize database with all required tables"""
    try:
        connection = DatabaseConfig.get_connection()
        if connection:
            cursor = connection.cursor()
            
            for table_name, schema in TABLE_SCHEMAS.items():
                cursor.execute(schema)
                print(f"✅ Table '{table_name}' initialized successfully")
            
            cursor.close()
            connection.close()
            print("✅ Database initialization completed")
            return True
        else:
            print("❌ Failed to initialize database - no connection")
            return False
            
    except Error as e:
        print(f"❌ Database initialization error: {e}")
        return False

if __name__ == "__main__":
    # Test database connection
    print("🔍 Testing database connection...")
    if DatabaseConfig.test_connection():
        print("✅ Database connection test passed")
        
        # Initialize database
        print("🔧 Initializing database...")
        if initialize_database():
            print("✅ Database setup completed successfully")
        else:
            print("❌ Database setup failed")
    else:
        print("❌ Database connection test failed")
