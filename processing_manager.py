import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
from supabase import create_client, Client
import os

class ProcessingSessionManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
    
    def create_session(self, file_name: str, file_size: int, file_hash: str, total_chunks_expected: int = 0) -> str:
        """Create a new processing session"""
        session_id = str(uuid.uuid4())
        
        try:
            result = self.supabase.table('processing_sessions').insert({
                'id': session_id,
                'file_name': file_name,
                'file_hash': file_hash,
                'file_size': file_size,
                'total_chunks_expected': total_chunks_expected,
                'chunks_uploaded': 0,
                'current_phase': 'initializing',
                'status': 'processing',
                'start_time': datetime.now().isoformat(),
                'metadata': {}
            }).execute()
            
            if result.data:
                return session_id
            else:
                raise Exception("Failed to create processing session")
                
        except Exception as e:
            print(f"Error creating processing session: {e}")
            raise e
    
    def update_session_progress(self, session_id: str, chunks_uploaded: int, current_phase: str, 
                              total_chunks_expected: Optional[int] = None, metadata: Optional[Dict] = None):
        """Update session progress"""
        try:
            update_data = {
                'chunks_uploaded': chunks_uploaded,
                'current_phase': current_phase,
                'updated_at': datetime.now().isoformat()
            }
            
            if total_chunks_expected is not None:
                update_data['total_chunks_expected'] = total_chunks_expected
            
            if metadata is not None:
                update_data['metadata'] = metadata
            
            result = self.supabase.table('processing_sessions').update(update_data).eq('id', session_id).execute()
            
            return result.data is not None
            
        except Exception as e:
            print(f"Error updating session progress: {e}")
            return False
    
    def complete_session(self, session_id: str, total_chunks: int):
        """Mark session as completed"""
        try:
            result = self.supabase.table('processing_sessions').update({
                'status': 'completed',
                'chunks_uploaded': total_chunks,
                'current_phase': 'completed',
                'end_time': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).eq('id', session_id).execute()
            
            return result.data is not None
            
        except Exception as e:
            print(f"Error completing session: {e}")
            return False
    
    def fail_session(self, session_id: str, error_message: str):
        """Mark session as failed"""
        try:
            result = self.supabase.table('processing_sessions').update({
                'status': 'failed',
                'current_phase': 'failed',
                'error_message': error_message,
                'end_time': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }).eq('id', session_id).execute()
            
            return result.data is not None
            
        except Exception as e:
            print(f"Error failing session: {e}")
            return False
    
    def create_checkpoint(self, session_id: str, checkpoint_phase: str, chunks_processed: int, 
                         current_batch: int, checkpoint_data: Dict, memory_usage_mb: int = 0):
        """Create a processing checkpoint"""
        try:
            result = self.supabase.table('processing_checkpoints').insert({
                'session_id': session_id,
                'checkpoint_phase': checkpoint_phase,
                'chunks_processed': chunks_processed,
                'current_batch': current_batch,
                'checkpoint_data': checkpoint_data,
                'memory_usage_mb': memory_usage_mb,
                'checkpoint_time': datetime.now().isoformat()
            }).execute()
            
            return result.data is not None
            
        except Exception as e:
            print(f"Error creating checkpoint: {e}")
            return False
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current session status"""
        try:
            result = self.supabase.table('processing_sessions').select('*').eq('id', session_id).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            print(f"Error getting session status: {e}")
            return None
    
    def get_all_processing_sessions(self) -> List[Dict]:
        """Get all processing sessions for display"""
        try:
            result = self.supabase.table('processing_sessions').select('*').order('start_time', desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            print(f"Error getting processing sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a processing session and its checkpoints"""
        try:
            # Delete checkpoints first (due to foreign key constraint)
            self.supabase.table('processing_checkpoints').delete().eq('session_id', session_id).execute()
            
            # Delete session
            result = self.supabase.table('processing_sessions').delete().eq('id', session_id).execute()
            
            return result.data is not None
            
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False