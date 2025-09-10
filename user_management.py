import sqlite3
import secrets
from datetime import datetime, timedelta
import uuid

class UserManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create users and sessions tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_code TEXT UNIQUE NOT NULL,
                client_name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                expires_at TIMESTAMP,
                subscription_tier TEXT DEFAULT 'basic'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_code TEXT NOT NULL,
                session_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (access_code) REFERENCES users (access_code)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_access_code(self, length=12):
        """Generate secure access code"""
        return secrets.token_urlsafe(length)[:length].upper()
    
    def add_user(self, client_name, email=None, subscription_tier='basic', days_valid=365):
        """Add new user with access code"""
        access_code = self.generate_access_code()
        expires_at = datetime.now() + timedelta(days=days_valid)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (access_code, client_name, email, expires_at, subscription_tier)
                VALUES (?, ?, ?, ?, ?)
            ''', (access_code, client_name, email, expires_at, subscription_tier))
            
            conn.commit()
            return access_code
        except sqlite3.IntegrityError:
            return self.add_user(client_name, email, subscription_tier, days_valid)
        finally:
            conn.close()
    
    def validate_access_code(self, access_code):
        """Check if access code is valid"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users 
            WHERE access_code = ? AND is_active = 1 AND expires_at > ?
        ''', (access_code, datetime.now()))
        
        user = cursor.fetchone()
        conn.close()
        
        return user is not None
    
    def create_session(self, access_code, session_id):
        """Create user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (access_code, session_id, last_activity)
            VALUES (?, ?, ?)
        ''', (access_code, session_id, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def is_session_valid(self, session_id, hours_timeout=24):
        """Check if session is valid (not expired)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timeout_threshold = datetime.now() - timedelta(hours=hours_timeout)
        
        cursor.execute('''
            SELECT us.*, u.is_active as user_active 
            FROM user_sessions us
            JOIN users u ON us.access_code = u.access_code
            WHERE us.session_id = ? AND us.is_active = 1 AND u.is_active = 1
            AND us.last_activity > ?
        ''', (session_id, timeout_threshold))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def update_session_activity(self, session_id):
        """Update session activity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_sessions 
            SET last_activity = ? 
            WHERE session_id = ? AND is_active = 1
        ''', (datetime.now(), session_id))
        
        conn.commit()
        conn.close()
    
    def deactivate_user(self, access_code):
        """Deactivate user and all their sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET is_active = 0 WHERE access_code = ?
        ''', (access_code,))
        
        cursor.execute('''
            UPDATE user_sessions SET is_active = 0 WHERE access_code = ?
        ''', (access_code,))
        
        conn.commit()
        conn.close()
    
    def get_all_users(self):
        """Get all users for admin panel"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT access_code, client_name, email, created_at, last_login, 
                   is_active, expires_at, subscription_tier
            FROM users 
            ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return users
    
    def update_last_login(self, access_code):
        """Update last login time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE access_code = ?
        ''', (datetime.now(), access_code))
        
        conn.commit()
        conn.close()
