/*
  # Create pending_responses table for background streaming

  1. New Tables
    - `pending_responses`
      - `id` (uuid, primary key) - Unique identifier for the pending response
      - `response_id` (text, unique, indexed) - OpenAI's response ID for retrieval/resume
      - `access_code` (text) - User's access code (references users table)
      - `session_id` (text, indexed) - Session identifier for recovery
      - `user_query` (text) - The original user query
      - `search_results` (jsonb) - RAG search results used for context
      - `reasoning_effort` (text) - "medium" or "high"
      - `status` (text) - "queued", "in_progress", "streaming", "completed", "failed"
      - `partial_response` (text) - Accumulated text during streaming (for recovery)
      - `final_response` (text) - Complete response when done
      - `token_usage` (jsonb) - Token counts (input, output, reasoning, total)
      - `error_message` (text) - Error details if failed
      - `sequence_number` (integer) - Last processed event sequence for resume
      - `time_to_first_token` (float) - Seconds until first text chunk
      - `created_at` (timestamptz) - When the request was created
      - `updated_at` (timestamptz) - Last update timestamp
      - `completed_at` (timestamptz) - When processing finished

  2. Security
    - Enable RLS on `pending_responses` table
    - Add policy for users to read/write their own pending responses

  3. Indexes
    - Index on session_id for fast recovery lookups
    - Index on status for cleanup queries
    - Unique index on response_id for OpenAI retrieval
*/

-- Create the pending_responses table
CREATE TABLE IF NOT EXISTS pending_responses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    response_id TEXT UNIQUE NOT NULL,
    access_code TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    search_results JSONB,
    reasoning_effort TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'queued',
    partial_response TEXT DEFAULT '',
    final_response TEXT,
    token_usage JSONB,
    error_message TEXT,
    sequence_number INTEGER DEFAULT 0,
    time_to_first_token FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_pending_responses_session ON pending_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_pending_responses_status ON pending_responses(status);
CREATE INDEX IF NOT EXISTS idx_pending_responses_access_code ON pending_responses(access_code);
CREATE INDEX IF NOT EXISTS idx_pending_responses_created ON pending_responses(created_at);

-- Enable Row Level Security
ALTER TABLE pending_responses ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own pending responses
CREATE POLICY "Users can read own pending responses"
    ON pending_responses
    FOR SELECT
    TO authenticated
    USING (access_code = current_setting('app.current_user_access_code', true));

-- Policy: Users can insert their own pending responses
CREATE POLICY "Users can insert own pending responses"
    ON pending_responses
    FOR INSERT
    TO authenticated
    WITH CHECK (access_code = current_setting('app.current_user_access_code', true));

-- Policy: Users can update their own pending responses
CREATE POLICY "Users can update own pending responses"
    ON pending_responses
    FOR UPDATE
    TO authenticated
    USING (access_code = current_setting('app.current_user_access_code', true))
    WITH CHECK (access_code = current_setting('app.current_user_access_code', true));

-- Policy: Service role can do everything (for backend operations)
CREATE POLICY "Service role has full access to pending responses"
    ON pending_responses
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Create function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_pending_responses_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for auto-updating updated_at
DROP TRIGGER IF EXISTS trigger_pending_responses_updated_at ON pending_responses;
CREATE TRIGGER trigger_pending_responses_updated_at
    BEFORE UPDATE ON pending_responses
    FOR EACH ROW
    EXECUTE FUNCTION update_pending_responses_updated_at();
