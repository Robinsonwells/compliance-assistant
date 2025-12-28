"""
Background Response Manager for handling OpenAI background streaming responses.
Manages pending responses in Supabase for recovery and tracking.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from supabase import create_client, Client


class BackgroundResponseManager:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )

    def create_pending_response(
        self,
        response_id: str,
        access_code: str,
        session_id: str,
        user_query: str,
        search_results: Optional[List[Dict]] = None,
        reasoning_effort: str = "medium"
    ) -> Optional[str]:
        try:
            data = {
                "response_id": response_id,
                "access_code": access_code,
                "session_id": session_id,
                "user_query": user_query,
                "search_results": json.dumps(search_results) if search_results else None,
                "reasoning_effort": reasoning_effort,
                "status": "queued",
                "partial_response": "",
                "sequence_number": 0
            }
            result = self.supabase.table("pending_responses").insert(data).execute()
            if result.data:
                return result.data[0]["id"]
            return None
        except Exception as e:
            print(f"Error creating pending response: {e}")
            return None

    def update_status(
        self,
        response_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        try:
            data = {"status": status}
            if error_message:
                data["error_message"] = error_message
            if status == "completed" or status == "failed":
                data["completed_at"] = datetime.now().isoformat()

            self.supabase.table("pending_responses")\
                .update(data)\
                .eq("response_id", response_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error updating status: {e}")
            return False

    def update_partial_response(
        self,
        response_id: str,
        partial_text: str,
        sequence_number: int
    ) -> bool:
        try:
            self.supabase.table("pending_responses")\
                .update({
                    "partial_response": partial_text,
                    "sequence_number": sequence_number,
                    "status": "streaming"
                })\
                .eq("response_id", response_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error updating partial response: {e}")
            return False

    def set_time_to_first_token(
        self,
        response_id: str,
        ttft: float
    ) -> bool:
        try:
            self.supabase.table("pending_responses")\
                .update({"time_to_first_token": ttft})\
                .eq("response_id", response_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error setting time to first token: {e}")
            return False

    def mark_completed(
        self,
        response_id: str,
        final_response: str,
        token_usage: Dict[str, int]
    ) -> bool:
        try:
            self.supabase.table("pending_responses")\
                .update({
                    "status": "completed",
                    "final_response": final_response,
                    "token_usage": json.dumps(token_usage),
                    "completed_at": datetime.now().isoformat()
                })\
                .eq("response_id", response_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error marking completed: {e}")
            return False

    def mark_failed(
        self,
        response_id: str,
        error_message: str
    ) -> bool:
        try:
            self.supabase.table("pending_responses")\
                .update({
                    "status": "failed",
                    "error_message": error_message,
                    "completed_at": datetime.now().isoformat()
                })\
                .eq("response_id", response_id)\
                .execute()
            return True
        except Exception as e:
            print(f"Error marking failed: {e}")
            return False

    def get_pending_response(self, response_id: str) -> Optional[Dict[str, Any]]:
        try:
            result = self.supabase.table("pending_responses")\
                .select("*")\
                .eq("response_id", response_id)\
                .maybeSingle()\
                .execute()
            return result.data
        except Exception as e:
            print(f"Error getting pending response: {e}")
            return None

    def get_active_response_for_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            result = self.supabase.table("pending_responses")\
                .select("*")\
                .eq("session_id", session_id)\
                .in_("status", ["queued", "in_progress", "streaming"])\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting active response: {e}")
            return None

    def get_recently_completed_response(
        self,
        session_id: str,
        minutes: int = 5
    ) -> Optional[Dict[str, Any]]:
        try:
            cutoff = (datetime.now() - timedelta(minutes=minutes)).isoformat()
            result = self.supabase.table("pending_responses")\
                .select("*")\
                .eq("session_id", session_id)\
                .eq("status", "completed")\
                .gte("completed_at", cutoff)\
                .order("completed_at", desc=True)\
                .limit(1)\
                .execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting recently completed response: {e}")
            return None

    def cleanup_old_responses(self, hours: int = 24) -> int:
        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
            result = self.supabase.table("pending_responses")\
                .delete()\
                .in_("status", ["completed", "failed"])\
                .lt("created_at", cutoff)\
                .execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"Error cleaning up old responses: {e}")
            return 0

    def get_user_pending_responses(
        self,
        access_code: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            result = self.supabase.table("pending_responses")\
                .select("*")\
                .eq("access_code", access_code)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting user pending responses: {e}")
            return []
