from supabase import create_client, Client
import secrets
from datetime import datetime, timedelta
import uuid
import os

class UserManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def generate_access_code(self, length=12):
        """Generate secure access code"""
        return secrets.token_urlsafe(length)[:length].upper()
    
    def add_user(self, client_name, email=None, subscription_tier='basic', days_valid=365):
        """Add new user with access code"""
        access_code = self.generate_access_code()
        expires_at = (datetime.now() + timedelta(days=days_valid)).isoformat()
        
        try:
            result = self.supabase.table('users').insert({
                'access_code': access_code,
                'client_name': client_name,
                'email': email,
                'expires_at': expires_at,
                'subscription_tier': subscription_tier
            }).execute()
            
            if result.data:
                return access_code
            else:
                # If insertion failed due to duplicate, try again
                return self.add_user(client_name, email, subscription_tier, days_valid)
        except Exception as e:
            # Handle duplicate access code by generating a new one
            if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                return self.add_user(client_name, email, subscription_tier, days_valid)
            raise e
    
    def validate_access_code(self, access_code):
        """Check if access code is valid"""
        try:
            result = self.supabase.table('users').select('*').eq('access_code', access_code).eq('is_active', True).execute()
            
            if not result.data:
                return False
            
            user = result.data[0]
            
            # Check if user has expired
            if user.get('expires_at'):
                expires_at = datetime.fromisoformat(user['expires_at'].replace('Z', '+00:00'))
                if expires_at < datetime.now(expires_at.tzinfo):
                    return False
            
            return True
        except Exception as e:
            print(f"Error validating access code: {e}")
            return False
    
    def create_session(self, access_code, session_id):
        """Create user session"""
        try:
            result = self.supabase.table('user_sessions').insert({
                'access_code': access_code,
                'session_id': session_id,
                'last_activity': datetime.now().isoformat()
            }).execute()
            
            return result.data is not None
        except Exception as e:
            print(f"Error creating session: {e}")
            return False
    
    def is_session_valid(self, session_id, hours_timeout=24):
        """Check if session is valid (not expired)"""
        try:
            timeout_threshold = (datetime.now() - timedelta(hours=hours_timeout)).isoformat()
            
            result = self.supabase.table('user_sessions').select('''
                *,
                users!inner(is_active)
            ''').eq('session_id', session_id).eq('is_active', True).gt('last_activity', timeout_threshold).execute()
            
            if not result.data:
                return False
            
            session = result.data[0]
            return session['users']['is_active'] == True
        except Exception as e:
            print(f"Error validating session: {e}")
            return False
    
    def update_session_activity(self, session_id):
        """Update session activity"""
        try:
            result = self.supabase.table('user_sessions').update({
                'last_activity': datetime.now().isoformat()
            }).eq('session_id', session_id).eq('is_active', True).execute()
            
            return result.data is not None
        except Exception as e:
            print(f"Error updating session activity: {e}")
            return False
    
    def deactivate_user(self, access_code):
        """Deactivate user and all their sessions"""
        try:
            # Deactivate user
            self.supabase.table('users').update({
                'is_active': False
            }).eq('access_code', access_code).execute()
            
            # Deactivate all sessions for this user
            self.supabase.table('user_sessions').update({
                'is_active': False
            }).eq('access_code', access_code).execute()
            
            return True
        except Exception as e:
            print(f"Error deactivating user: {e}")
            return False
    
    def get_all_users(self):
        """Get all users for admin panel"""
        try:
            result = self.supabase.table('users').select(
                'access_code, client_name, email, created_at, last_login, is_active, expires_at, subscription_tier'
            ).order('created_at', desc=True).execute()
            
            if result.data:
                # Convert to tuple format to match original SQLite implementation
                users = []
                for user in result.data:
                    users.append((
                        user['access_code'],
                        user['client_name'],
                        user['email'],
                        user['created_at'],
                        user['last_login'],
                        user['is_active'],
                        user['expires_at'],
                        user['subscription_tier']
                    ))
                return users
            return []
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
    
    def update_last_login(self, access_code):
        """Update last login time"""
        try:
            result = self.supabase.table('users').update({
                'last_login': datetime.now().isoformat()
            }).eq('access_code', access_code).execute()
            
            return result.data is not None
        except Exception as e:
            print(f"Error updating last login: {e}")
            return False