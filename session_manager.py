"""
Processing Session Manager
Handles Supabase database interactions for session tracking and checkpoints
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class SessionManager:
    """Manage processing sessions and checkpoints in Supabase"""

    def __init__(self):
        """Initialize Supabase client"""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_KEY) must be set")

        self.client: Client = create_client(supabase_url, supabase_key)

    def create_session(
        self,
        file_name: str,
        file_hash: str,
        file_size: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new processing session"""
        try:
            data = {
                'file_name': file_name,
                'file_hash': file_hash,
                'file_size': file_size,
                'status': 'processing',
                'current_phase': 'initializing',
                'metadata': metadata or {}
            }

            result = self.client.table('processing_sessions').insert(data).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            else:
                raise Exception("Failed to create session")

        except Exception as e:
            raise Exception(f"Error creating session: {str(e)}")

    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing processing session"""
        try:
            self.client.table('processing_sessions').update(updates).eq('id', session_id).execute()
            return True
        except Exception as e:
            print(f"Error updating session: {e}")
            return False

    def update_session_phase(
        self,
        session_id: str,
        phase: str,
        chunks_uploaded: Optional[int] = None
    ) -> bool:
        """Update the current phase of a session"""
        updates = {'current_phase': phase}

        if chunks_uploaded is not None:
            updates['chunks_uploaded'] = chunks_uploaded

        return self.update_session(session_id, updates)

    def complete_session(
        self,
        session_id: str,
        total_chunks: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> bool:
        """Mark a session as completed or failed"""
        updates = {
            'status': 'completed' if success else 'failed',
            'end_time': datetime.now().isoformat(),
            'total_chunks_expected': total_chunks,
            'chunks_uploaded': total_chunks if success else 0
        }

        if error_message:
            updates['error_message'] = error_message

        return self.update_session(session_id, updates)

    def create_checkpoint(
        self,
        session_id: str,
        phase: str,
        chunks_processed: int,
        current_batch: int,
        checkpoint_data: Optional[Dict] = None,
        memory_usage_mb: Optional[int] = None
    ) -> Optional[str]:
        """Create a checkpoint for resume capability"""
        try:
            data = {
                'session_id': session_id,
                'checkpoint_phase': phase,
                'chunks_processed': chunks_processed,
                'current_batch': current_batch,
                'checkpoint_data': checkpoint_data or {},
                'memory_usage_mb': memory_usage_mb or 0
            }

            result = self.client.table('processing_checkpoints').insert(data).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            else:
                return None

        except Exception as e:
            print(f"Error creating checkpoint: {e}")
            return None

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details"""
        try:
            result = self.client.table('processing_sessions').select('*').eq('id', session_id).execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            print(f"Error getting session: {e}")
            return None

    def get_session_by_hash(self, file_hash: str) -> Optional[Dict]:
        """Get most recent session for a file hash"""
        try:
            result = self.client.table('processing_sessions')\
                .select('*')\
                .eq('file_hash', file_hash)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            print(f"Error getting session by hash: {e}")
            return None

    def get_latest_checkpoint(self, session_id: str) -> Optional[Dict]:
        """Get the most recent checkpoint for a session"""
        try:
            result = self.client.table('processing_checkpoints')\
                .select('*')\
                .eq('session_id', session_id)\
                .order('checkpoint_time', desc=True)\
                .limit(1)\
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            print(f"Error getting latest checkpoint: {e}")
            return None

    def get_all_checkpoints(self, session_id: str) -> List[Dict]:
        """Get all checkpoints for a session"""
        try:
            result = self.client.table('processing_checkpoints')\
                .select('*')\
                .eq('session_id', session_id)\
                .order('checkpoint_time', desc=False)\
                .execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Error getting checkpoints: {e}")
            return []

    def get_active_sessions(self) -> List[Dict]:
        """Get all currently active processing sessions"""
        try:
            result = self.client.table('processing_sessions')\
                .select('*')\
                .eq('status', 'processing')\
                .order('start_time', desc=True)\
                .execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Error getting active sessions: {e}")
            return []

    def get_completed_sessions(self, limit: int = 50) -> List[Dict]:
        """Get recently completed sessions"""
        try:
            result = self.client.table('processing_sessions')\
                .select('*')\
                .eq('status', 'completed')\
                .order('end_time', desc=True)\
                .limit(limit)\
                .execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Error getting completed sessions: {e}")
            return []

    def delete_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days"""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()

            result = self.client.table('processing_sessions')\
                .delete()\
                .lt('created_at', cutoff_iso)\
                .execute()

            return len(result.data) if result.data else 0

        except Exception as e:
            print(f"Error deleting old sessions: {e}")
            return 0

    def check_duplicate_by_hash(self, file_hash: str) -> bool:
        """Check if a file with this hash has been successfully processed"""
        try:
            result = self.client.table('processing_sessions')\
                .select('id')\
                .eq('file_hash', file_hash)\
                .eq('status', 'completed')\
                .limit(1)\
                .execute()

            return result.data and len(result.data) > 0

        except Exception as e:
            print(f"Error checking duplicate: {e}")
            return False
