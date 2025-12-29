# Setup Guide - Cloudflare-Safe Architecture

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL database (Supabase)
- OpenAI API key with GPT-5 access
- Qdrant instance for vector search

### Local Development Setup

1. **Clone and Install Dependencies**

```bash
# Install Python dependencies
pip install -r requirements.txt
```

2. **Configure Environment Variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or your preferred editor
```

Required variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Service role key (for backend operations)
- `OPENAI_API_KEY` - OpenAI API key
- `QDRANT_URL` - Qdrant instance URL
- `QDRANT_API_KEY` - Qdrant API key
- `FASTAPI_URL` - Set to `http://localhost:8000` for local dev

3. **Run Database Migrations**

The `pending_responses` table should already exist. If not:

```bash
# Connect to your Supabase project and run:
# supabase/migrations/20251228080415_create_pending_responses_table.sql
```

4. **Start the Servers**

**Option A: Use the startup script (recommended)**

```bash
chmod +x start_servers.sh
bash start_servers.sh
```

**Option B: Run servers separately**

Terminal 1 - FastAPI Backend:
```bash
uvicorn fastapi_backend:app --reload --port 8000
```

Terminal 2 - Streamlit Frontend:
```bash
streamlit run app.py
```

5. **Access the Application**

- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- FastAPI health: http://localhost:8000/health

## Testing the Implementation

### Manual Testing

1. Open Streamlit at http://localhost:8501
2. Login with valid access code
3. Submit a complex legal query (e.g., "What are California's data privacy requirements?")
4. Watch the status updates in real-time
5. Verify the response completes without timeout

### Check Backend Logs

FastAPI logs will show:
```
[BACKGROUND] Started processing response_id: abc123...
[BACKGROUND] First token received in 3.5s
[BACKGROUND] Streaming completed in 180s
[BACKGROUND] Response abc123 marked as completed
```

### Check Database

Query Supabase to see response records:

```sql
SELECT
  response_id,
  status,
  time_to_first_token,
  created_at,
  updated_at,
  completed_at
FROM pending_responses
ORDER BY created_at DESC
LIMIT 10;
```

### Verify Polling Behavior

Open browser DevTools → Network tab:
- You should see repeated calls to `/api/response/{id}`
- Each call should complete in <100ms
- Calls should happen every 0.5-2 seconds

## Production Deployment

### Option 1: Heroku (Single Platform)

1. **Create Heroku App**

```bash
heroku create your-app-name
```

2. **Set Environment Variables**

```bash
heroku config:set OPENAI_API_KEY=your-key
heroku config:set SUPABASE_URL=your-url
heroku config:set SUPABASE_SERVICE_KEY=your-key
heroku config:set QDRANT_URL=your-url
heroku config:set QDRANT_API_KEY=your-key
heroku config:set FASTAPI_URL=https://your-app-name.herokuapp.com
```

3. **Deploy**

```bash
git push heroku main
```

The `Procfile` will automatically start both servers.

### Option 2: Separate Services

**Deploy FastAPI Backend** (to Cloud Run, ECS, Fly.io, etc.):

```bash
# Example: Google Cloud Run
gcloud run deploy fastapi-backend \
  --source . \
  --command "uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000" \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,SUPABASE_URL=$SUPABASE_URL"
```

**Deploy Streamlit Frontend** (to Streamlit Cloud):

1. Connect GitHub repo to Streamlit Cloud
2. Set environment variables in Streamlit Cloud dashboard
3. Set `FASTAPI_URL` to your FastAPI backend URL
4. Deploy

### Option 3: Docker Compose

```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

## Troubleshooting

### FastAPI won't start

**Error**: `Port 8000 already in use`

**Solution**: Kill existing process or use different port:
```bash
lsof -ti:8000 | xargs kill -9
# or
FASTAPI_PORT=8001 uvicorn fastapi_backend:app --port 8001
```

### Streamlit can't connect to FastAPI

**Error**: `Connection refused` or `Failed to fetch`

**Solutions**:
1. Verify FastAPI is running: `curl http://localhost:8000/health`
2. Check `FASTAPI_URL` in `.env`
3. Ensure no firewall blocking localhost

### Database connection errors

**Error**: `Failed to create pending response`

**Solutions**:
1. Verify Supabase credentials in `.env`
2. Check `pending_responses` table exists
3. Verify RLS policies allow service role access

### OpenAI API errors

**Error**: `AuthenticationError` or `InvalidRequestError`

**Solutions**:
1. Verify `OPENAI_API_KEY` is valid
2. Check OpenAI account has GPT-5 access
3. Verify API usage limits not exceeded

### Responses stuck in "queued"

**Possible Causes**:
1. Background task failed silently
2. OpenAI API error
3. Database write failed

**Debug Steps**:
1. Check FastAPI logs for errors
2. Query database: `SELECT * FROM pending_responses WHERE status='queued'`
3. Manually mark as failed: `UPDATE pending_responses SET status='failed' WHERE response_id='xxx'`

### Cloudflare timeouts still happening

**This should not happen with new architecture!**

**Debug Steps**:
1. Verify you're calling `generate_legal_response_via_fastapi()` not old functions
2. Check FastAPI logs - background task should be running
3. Verify polling is happening (check Network tab)
4. Ensure each poll completes in <1s

If timeouts persist, the issue is likely NOT in this architecture but in:
- Web server configuration
- Reverse proxy settings
- Client-side network issues

## Monitoring

### Health Checks

```bash
# FastAPI health
curl http://localhost:8000/health

# Check active responses
curl http://localhost:8000/api/response/{response_id}
```

### Database Queries

```sql
-- Active responses
SELECT COUNT(*)
FROM pending_responses
WHERE status IN ('queued', 'in_progress', 'streaming');

-- Recent completions
SELECT COUNT(*), AVG(time_to_first_token)
FROM pending_responses
WHERE status = 'completed'
  AND created_at > NOW() - INTERVAL '1 hour';

-- Failed responses
SELECT response_id, error_message, created_at
FROM pending_responses
WHERE status = 'failed'
ORDER BY created_at DESC
LIMIT 10;
```

### Cleanup Old Responses

```sql
-- Delete old completed/failed responses (>24 hours)
DELETE FROM pending_responses
WHERE status IN ('completed', 'failed')
  AND created_at < NOW() - INTERVAL '24 hours';
```

Or use the built-in cleanup:
```python
from background_response_manager import BackgroundResponseManager
manager = BackgroundResponseManager()
deleted = manager.cleanup_old_responses(hours=24)
print(f"Deleted {deleted} old responses")
```

## Performance Tuning

### Reduce Polling Frequency

In `app.py`, adjust intervals:

```python
# Current (aggressive for demo):
interval = 0.5 if status == "streaming" else 1.0 if status == "in_progress" else 2.0

# More conservative (reduces DB load):
interval = 1.0 if status == "streaming" else 2.0 if status == "in_progress" else 3.0
```

### Optimize Partial Response Storage

In `openai_background_processor.py`, adjust update frequency:

```python
# Current (every 50 chunks):
if seq % 50 == 0:
    manager.update_partial_response(...)

# Less frequent (every 100 chunks):
if seq % 100 == 0:
    manager.update_partial_response(...)
```

### Database Indexing

Ensure indexes exist:

```sql
-- Should already exist, but verify:
CREATE INDEX IF NOT EXISTS idx_pending_responses_session
  ON pending_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_pending_responses_status
  ON pending_responses(status);
```

## Migration from Old Approach

If you're migrating from the old `generate_legal_response_polling()` approach:

1. **Update Environment**: Add `FASTAPI_URL` to `.env`
2. **Deploy FastAPI**: Start the backend server
3. **Update Code**: Replace calls to old functions with `generate_legal_response_via_fastapi()`
4. **Test**: Verify no timeouts occur
5. **Monitor**: Watch for 1-2 days
6. **Cleanup**: Remove old polling code (optional)

The new approach is a drop-in replacement with identical signature:

```python
# Old way (still works but may timeout):
result = generate_legal_response_smart(...)

# New way (Cloudflare-safe):
result = generate_legal_response_via_fastapi(...)
```

## FAQ

**Q: Do I need to run both servers for development?**
A: Yes, Streamlit polls the FastAPI backend, so both must be running.

**Q: Can I use this with other LLM providers?**
A: Yes! Modify `openai_background_processor.py` to call Anthropic, Cohere, etc.

**Q: What if my FastAPI backend crashes?**
A: Streamlit will show connection errors. Restart FastAPI and responses will resume polling.

**Q: How much does this cost to run?**
A: Same LLM costs as before. Additional costs:
- Supabase: Free tier covers most usage
- Hosting: Depends on platform (Heroku $7/month for hobby dyno)

**Q: Can multiple users submit queries simultaneously?**
A: Yes! Each gets independent background task. Limited only by server resources.

**Q: What's the maximum processing time?**
A: No hard limit in the architecture. Limited by:
- OpenAI's maximum generation time
- Your server's resources
- Practical user patience

## Next Steps

1. Complete local setup and testing
2. Deploy to staging environment
3. Monitor performance and costs
4. Consider enhancements:
   - WebSocket instead of polling
   - Response caching
   - Priority queue
   - Multi-model support

## Support

For issues or questions:
1. Check logs (FastAPI + Streamlit)
2. Query database for stuck responses
3. Review this guide's troubleshooting section
4. Check `CLOUDFLARE_SAFE_ARCHITECTURE.md` for architecture details
