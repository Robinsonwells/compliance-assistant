/*
  # Create Processing Sessions and Checkpoints Tables

  1. New Tables
    - `processing_sessions`
      - `id` (uuid, primary key) - Unique session identifier
      - `file_name` (text) - Original filename
      - `file_hash` (text) - MD5 hash of file content for deduplication
      - `file_size` (bigint) - File size in bytes
      - `total_chunks_expected` (integer) - Estimated total chunks to process
      - `chunks_uploaded` (integer) - Number of chunks successfully uploaded
      - `current_phase` (text) - Current processing phase
      - `status` (text) - Session status: processing, completed, failed, paused
      - `start_time` (timestamptz) - When processing started
      - `end_time` (timestamptz) - When processing completed/failed
      - `error_message` (text) - Error details if failed
      - `metadata` (jsonb) - Additional metadata (document type, jurisdiction, etc.)
      - `created_at` (timestamptz) - Record creation time
      - `updated_at` (timestamptz) - Last update time
    
    - `processing_checkpoints`
      - `id` (uuid, primary key) - Unique checkpoint identifier
      - `session_id` (uuid, foreign key) - References processing_sessions
      - `checkpoint_phase` (text) - Phase at checkpoint (chunking, embedding, upload)
      - `chunks_processed` (integer) - Chunks processed at this checkpoint
      - `current_batch` (integer) - Current batch number
      - `checkpoint_data` (jsonb) - Serialized state data for resume
      - `checkpoint_time` (timestamptz) - When checkpoint was created
      - `memory_usage_mb` (integer) - Memory usage at checkpoint
      
  2. Indexes
    - Index on file_hash for quick duplicate detection
    - Index on session_id in checkpoints for fast lookups
    - Index on status for filtering active sessions
    
  3. Security
    - Enable RLS on both tables
    - Policies allow authenticated admin users to manage all records
*/

-- Create processing_sessions table
CREATE TABLE IF NOT EXISTS processing_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  file_name text NOT NULL,
  file_hash text NOT NULL,
  file_size bigint NOT NULL DEFAULT 0,
  total_chunks_expected integer DEFAULT 0,
  chunks_uploaded integer DEFAULT 0,
  current_phase text DEFAULT 'initializing',
  status text DEFAULT 'processing',
  start_time timestamptz DEFAULT now(),
  end_time timestamptz,
  error_message text,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create processing_checkpoints table
CREATE TABLE IF NOT EXISTS processing_checkpoints (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES processing_sessions(id) ON DELETE CASCADE,
  checkpoint_phase text NOT NULL,
  chunks_processed integer DEFAULT 0,
  current_batch integer DEFAULT 0,
  checkpoint_data jsonb DEFAULT '{}'::jsonb,
  checkpoint_time timestamptz DEFAULT now(),
  memory_usage_mb integer DEFAULT 0
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_processing_sessions_file_hash 
  ON processing_sessions(file_hash);

CREATE INDEX IF NOT EXISTS idx_processing_sessions_status 
  ON processing_sessions(status);

CREATE INDEX IF NOT EXISTS idx_processing_checkpoints_session_id 
  ON processing_checkpoints(session_id);

-- Enable Row Level Security
ALTER TABLE processing_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_checkpoints ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users (admin access)
CREATE POLICY "Authenticated users can view all processing sessions"
  ON processing_sessions
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can insert processing sessions"
  ON processing_sessions
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update processing sessions"
  ON processing_sessions
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete processing sessions"
  ON processing_sessions
  FOR DELETE
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can view all checkpoints"
  ON processing_checkpoints
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can insert checkpoints"
  ON processing_checkpoints
  FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update checkpoints"
  ON processing_checkpoints
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete checkpoints"
  ON processing_checkpoints
  FOR DELETE
  TO authenticated
  USING (true);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for processing_sessions
CREATE TRIGGER update_processing_sessions_updated_at
  BEFORE UPDATE ON processing_sessions
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();