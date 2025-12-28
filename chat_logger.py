"""
Chat logging utilities for Supabase
"""
import os
from datetime import datetime
from supabase import create_client, Client
from typing import Optional

class ChatLogger:
    def __init__(self):
        """Initialize Supabase client for logging"""
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY")
        )
    
    def log_chat(
        self,
        access_code: str,
        user_query: str,
        gpt5_response: Optional[str] = None,
        perplexity_response: Optional[str] = None,
        session_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        main_model_used: Optional[str] = None,
        response_mode: Optional[str] = None,
        time_to_first_token: Optional[float] = None,
        total_streaming_time: Optional[float] = None,
        was_recovered: bool = False,
        response_id: Optional[str] = None
    ) -> bool:
        """
        Log a chat interaction to Supabase

        Args:
            access_code: User's access code
            user_query: The question the user asked
            gpt5_response: GPT-5's answer (optional)
            perplexity_response: Perplexity's audit response (optional)
            session_id: Session ID (optional)
            reasoning_effort: "medium" or "high" (optional)
            tokens_used: Total tokens consumed (optional)
            cost_estimate: Estimated cost in dollars (optional)
            main_model_used: Model used for main response generation (optional)
            response_mode: "streaming_background", "streaming", "sync", or "background" (optional)
            time_to_first_token: Seconds until first text chunk arrived (optional)
            total_streaming_time: Total time from start to completion (optional)
            was_recovered: Whether response was recovered after timeout (optional)
            response_id: OpenAI response ID for background requests (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            MAX_LENGTH = 50000

            if gpt5_response and len(gpt5_response) > MAX_LENGTH:
                gpt5_response = gpt5_response[:MAX_LENGTH] + "... [TRUNCATED]"

            if perplexity_response and len(perplexity_response) > MAX_LENGTH:
                perplexity_response = perplexity_response[:MAX_LENGTH] + "... [TRUNCATED]"

            data = {
                "access_code": access_code,
                "user_query": user_query,
                "gpt5_response": gpt5_response,
                "perplexity_response": perplexity_response,
                "session_id": session_id,
                "reasoning_effort": reasoning_effort,
                "tokens_used": tokens_used,
                "cost_estimate": cost_estimate,
                "main_model_used": main_model_used
            }

            if response_mode:
                data["response_mode"] = response_mode
            if time_to_first_token is not None:
                data["time_to_first_token"] = time_to_first_token
            if total_streaming_time is not None:
                data["total_streaming_time"] = total_streaming_time
            if was_recovered:
                data["was_recovered"] = was_recovered
            if response_id:
                data["response_id"] = response_id

            response = self.supabase.table("chat_logs").insert(data).execute()
            return True
        except Exception as e:
            print(f"Error logging chat: {e}")
            return False
    
    def get_logs_by_access_code(
        self,
        access_code: str,
        limit: int = 100
    ):
        """
        Retrieve chat logs for a specific access code
        
        Args:
            access_code: User's access code
            limit: Maximum number of logs to retrieve
        
        Returns:
            List of chat logs, most recent first
        """
        try:
            response = self.supabase.table("chat_logs")\
                .select("*")\
                .eq("access_code", access_code)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return response.data
        except Exception as e:
            print(f"Error retrieving logs: {e}")
            return []
    
    def get_all_logs(
        self,
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Retrieve all chat logs (for admin dashboard)
        
        Args:
            limit: Maximum number of logs to retrieve
            start_date: Filter logs after this date (optional)
            end_date: Filter logs before this date (optional)
        
        Returns:
            List of chat logs, most recent first
        """
        try:
            query = self.supabase.table("chat_logs").select("*")
            
            if start_date:
                query = query.gte("created_at", start_date.isoformat())
            if end_date:
                query = query.lte("created_at", end_date.isoformat())
            
            response = query.order("created_at", desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"Error retrieving all logs: {e}")
            return []