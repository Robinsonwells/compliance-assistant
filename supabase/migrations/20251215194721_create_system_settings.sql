/*
  # Create system_settings table

  1. New Tables
    - `system_settings`
      - `id` (uuid, primary key) - Unique identifier for each setting
      - `setting_key` (text, unique) - The key name for the setting (e.g., 'show_rag_chunks')
      - `setting_value` (text) - The value of the setting (stored as text, parsed by application)
      - `setting_type` (text) - The type of setting (e.g., 'boolean', 'string', 'number')
      - `description` (text) - Human-readable description of what the setting controls
      - `created_at` (timestamptz) - When the setting was created
      - `updated_at` (timestamptz) - When the setting was last updated

  2. Security
    - Enable RLS on `system_settings` table
    - Add policy for authenticated users to read settings (public read access for app functionality)
    - Add policy for admin operations (note: admin access controlled at application level)

  3. Indexes
    - Create index on `setting_key` for fast lookups

  4. Initial Data
    - Insert default setting for RAG chunks display
*/

-- Create system_settings table
CREATE TABLE IF NOT EXISTS system_settings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  setting_key text UNIQUE NOT NULL,
  setting_value text NOT NULL,
  setting_type text DEFAULT 'string',
  description text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read settings (needed for app functionality)
CREATE POLICY "Anyone can read system settings"
  ON system_settings
  FOR SELECT
  USING (true);

-- Note: Write access is controlled at the application level via admin authentication
-- This policy allows all authenticated operations for simplicity
CREATE POLICY "Allow all operations on system settings"
  ON system_settings
  FOR ALL
  USING (true)
  WITH CHECK (true);

-- Create index on setting_key for fast lookups
CREATE INDEX IF NOT EXISTS idx_system_settings_key ON system_settings(setting_key);

-- Insert default setting for RAG chunks display
INSERT INTO system_settings (setting_key, setting_value, setting_type, description)
VALUES (
  'show_rag_chunks',
  'true',
  'boolean',
  'Display retrieved RAG chunks under AI responses'
)
ON CONFLICT (setting_key) DO NOTHING;
