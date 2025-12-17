from supabase import create_client, Client
from datetime import datetime
import os
from typing import Optional, Dict, Any

class SettingsManager:
    """Manage system-wide settings stored in Supabase"""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_timeout_seconds = 300  # 5 minutes

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False

        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_timeout_seconds

    def _refresh_cache(self):
        """Refresh settings cache from database"""
        try:
            result = self.supabase.table('system_settings').select('*').execute()

            if result.data:
                self._cache = {
                    row['setting_key']: row['setting_value']
                    for row in result.data
                }
                self._cache_timestamp = datetime.now()
        except Exception as e:
            print(f"Error refreshing settings cache: {e}")
            # Keep existing cache on error

    def get_setting(self, key: str, default: str = 'true') -> str:
        """
        Get a setting value by key, ALWAYS returns lowercase 'true' or 'false'
        Handles: bool, int (0/1), string variants, None, corrupted data
        """
        try:
            # Check cache first
            if not self._is_cache_valid():
                self._refresh_cache()

            # Get raw value from cache or database
            raw_value = None
            if key in self._cache:
                raw_value = self._cache[key]
            else:
                # Fallback to direct database query
                result = self.supabase.table('system_settings').select('setting_value').eq('setting_key', key).execute()
                if result.data:
                    raw_value = result.data[0]['setting_value']
                    self._cache[key] = raw_value

            # Normalize value to canonical 'true' or 'false'
            normalized = self._normalize_boolean_value(raw_value, default)

            # Update cache with normalized value
            if key in self._cache:
                self._cache[key] = normalized

            return normalized

        except Exception as e:
            print(f"Error getting setting '{key}': {e}")
            return self._normalize_boolean_value(default, 'true')

    def get_enum_setting(self, key: str, default: str, allowed_values: list) -> str:
        """
        Get an enum setting value by key
        Returns the raw string value if valid, otherwise returns default
        """
        try:
            if not self._is_cache_valid():
                self._refresh_cache()

            raw_value = None
            if key in self._cache:
                raw_value = self._cache[key]
            else:
                result = self.supabase.table('system_settings').select('setting_value').eq('setting_key', key).execute()
                if result.data:
                    raw_value = result.data[0]['setting_value']
                    self._cache[key] = raw_value

            if raw_value is None:
                return default

            value_lower = raw_value.lower().strip()
            if value_lower in [v.lower() for v in allowed_values]:
                return value_lower

            print(f"Invalid enum value '{raw_value}' for '{key}', using default '{default}'")
            return default

        except Exception as e:
            print(f"Error getting enum setting '{key}': {e}")
            return default

    def _normalize_boolean_value(self, value: Any, default: str = 'true') -> str:
        """
        Convert any value to canonical 'true' or 'false' string
        Handles: bool, int (0/1), various string formats, None, corrupted data
        """
        if value is None:
            return self._normalize_boolean_value(default, 'true')

        # Handle boolean type
        if isinstance(value, bool):
            return 'true' if value else 'false'

        # Handle integer (0/1 from some databases)
        if isinstance(value, int):
            return 'true' if value == 1 else 'false'

        # Handle string variants
        if isinstance(value, str):
            value_lower = value.lower().strip()

            # Truthy values
            if value_lower in ('true', '1', 'yes', 't', 'y', 'on', 'enabled'):
                return 'true'

            # Falsy values
            if value_lower in ('false', '0', 'no', 'f', 'n', 'off', 'disabled'):
                return 'false'

            # Corrupted/unexpected value - log warning
            print(f"⚠️ WARNING: Unexpected setting value '{value}', using default '{default}'")
            return self._normalize_boolean_value(default, 'true')

        # Unknown type - log warning
        print(f"⚠️ WARNING: Unknown setting type {type(value).__name__}, using default '{default}'")
        return self._normalize_boolean_value(default, 'true')

    def update_setting(self, key: str, value: str) -> bool:
        """
        Update a setting value
        Returns True if successful, False otherwise
        """
        try:
            # Check if setting exists
            result = self.supabase.table('system_settings').select('id').eq('setting_key', key).execute()

            if result.data:
                # Update existing setting
                update_result = self.supabase.table('system_settings').update({
                    'setting_value': value,
                    'updated_at': datetime.now().isoformat()
                }).eq('setting_key', key).execute()

                if update_result.data:
                    # Update cache
                    self._cache[key] = value
                    return True
            else:
                # Insert new setting if it doesn't exist
                insert_result = self.supabase.table('system_settings').insert({
                    'setting_key': key,
                    'setting_value': value,
                    'setting_type': 'boolean',
                    'description': f'Setting for {key}'
                }).execute()

                if insert_result.data:
                    # Update cache
                    self._cache[key] = value
                    return True

            return False

        except Exception as e:
            print(f"Error updating setting '{key}': {e}")
            return False

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        try:
            # Refresh cache
            self._refresh_cache()
            return self._cache.copy()

        except Exception as e:
            print(f"Error getting all settings: {e}")
            return {}

    def clear_cache(self):
        """Clear the settings cache to force refresh"""
        self._cache = {}
        self._cache_timestamp = None

    def get_setting_details(self, key: str) -> Optional[Dict[str, Any]]:
        """Get full details of a setting including metadata"""
        try:
            result = self.supabase.table('system_settings').select('*').eq('setting_key', key).execute()

            if result.data:
                return result.data[0]

            return None

        except Exception as e:
            print(f"Error getting setting details for '{key}': {e}")
            return None

    def initialize_default_settings(self):
        """Initialize default settings if they don't exist"""
        default_settings = [
            {
                'setting_key': 'show_rag_chunks',
                'setting_value': 'false',
                'setting_type': 'boolean',
                'description': 'Display retrieved RAG chunks under AI responses'
            },
            {
                'setting_key': 'default_reasoning_effort',
                'setting_value': 'automatic',
                'setting_type': 'enum',
                'description': 'Default reasoning effort level: automatic, medium, or high'
            }
        ]

        for setting in default_settings:
            try:
                # Check if setting already exists
                result = self.supabase.table('system_settings').select('id').eq('setting_key', setting['setting_key']).execute()

                if not result.data:
                    # Insert new default setting
                    self.supabase.table('system_settings').insert(setting).execute()
                    print(f"Initialized default setting: {setting['setting_key']}")

            except Exception as e:
                print(f"Error initializing default setting '{setting['setting_key']}': {e}")

        # Verify all default settings were initialized correctly
        print("=" * 50)
        print("📋 Settings Initialization Summary:")
        for setting in default_settings:
            key = setting['setting_key']
            try:
                current_value = self.get_setting(key, setting['setting_value'])
                print(f"  ✅ {key}: {current_value}")
            except Exception as e:
                print(f"  ❌ {key}: ERROR - {e}")
        print("=" * 50)
