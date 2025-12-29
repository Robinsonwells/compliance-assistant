# Cloudflare-Safe Architecture Documentation

## Overview

This document describes the Cloudflare-safe, long-running LLM pipeline implementation that prevents HTTP timeouts while maintaining near real-time user experience.

## Problem Statement

**Challenge**: LLM responses (GPT-5 with reasoning) can take 150-300+ seconds, but Cloudflare imposes a ~100s timeout on HTTP requests.

**Previous Approach**: Streamlit polled OpenAI's Responses API directly, but the HTTP connection remained open during the entire polling loop, causing timeouts.

**Solution**: Implement the Asynchronous Request-Reply pattern with a FastAPI backend that decouples LLM processing from HTTP requests entirely.

## Architecture Components

### 1. Database Layer (Supabase)

**Table**: `pending_responses`

**Purpose**: Persistent state store for tracking background LLM processing

**Key Fields**:
- `response_id` (TEXT, UNIQUE) - Globally unique identifier for each response
- `status` (TEXT) - Current state: `queued` → `in_progress` → `streaming` → `completed`/`failed`
- `partial_response` (TEXT) - Incremental streaming snapshot
- `final_response` (TEXT) - Complete response when done
- `sequence_number` (INT) - Monotonically increasing counter for streaming updates
- `time_to_first_token` (FLOAT) - Performance metric
- `token_usage` (JSONB) - Token counts and cost data
- `error_message` (TEXT) - Error details if failed

**Indexes**:
- `response_id` - Fast lookups by ID
- `session_id` - Recovery lookups
- `status` - Cleanup queries

### 2. BackgroundResponseManager (Python)

**File**: `background_response_manager.py`

**Purpose**: Abstraction layer for all Supabase operations

**Key Methods**:
- `create_pending_response()` - Initialize new response record
- `update_status()` - Transition status states
- `update_partial_response()` - Write streaming progress
- `set_time_to_first_token()` - Record performance metrics
- `mark_completed()` - Finalize successful response
- `mark_failed()` - Handle errors
- `get_pending_response()` - Retrieve current state

**Invariants Maintained**:
- `response_id` is globally unique
- Status transitions follow allowed paths
- `updated_at` is set on every write
- `sequence_number` monotonically increases

### 3. FastAPI Backend

**File**: `fastapi_backend.py`

**Purpose**: HTTP API that enables async request-reply pattern

**Endpoints**:

#### POST /api/chat
**Purpose**: Submit new query (returns immediately)

**Flow**:
1. Accept query, access_code, session_id, reasoning_effort, search_results
2. Generate unique `response_id`
3. Create `pending_responses` record with status='queued'
4. Spawn background task (DO NOT await it)
5. Return `response_id` to client (< 1000ms)

**Response**:
```json
{
  "response_id": "uuid",
  "status": "queued",
  "message": "Your query is being processed."
}
```

#### GET /api/response/{response_id}
**Purpose**: Poll for current status (high-frequency, lightweight)

**Flow**:
1. Read `pending_responses` record by `response_id`
2. Return current status and partial response
3. Complete in tens of milliseconds (single DB read)

**Response**:
```json
{
  "response_id": "uuid",
  "status": "streaming",
  "partial_response": "partial text...",
  "sequence_number": 42,
  "time_to_first_token": 3.5,
  "error_message": null
}
```

#### GET /api/response/{response_id}/final
**Purpose**: Retrieve final result after completion

**Flow**:
1. Read `pending_responses` record
2. Verify status=='completed'
3. Return final response and token usage

**Response**:
```json
{
  "response_id": "uuid",
  "status": "completed",
  "text": "complete response...",
  "token_usage": {
    "input_tokens": 1000,
    "output_tokens": 2000,
    "reasoning_tokens": 500,
    "total_tokens": 3500
  },
  "time_to_first_token": 3.5
}
```

### 4. Background OpenAI Processor

**File**: `openai_background_processor.py`

**Purpose**: Handle long-running LLM calls completely decoupled from HTTP

**Function**: `process_openai_background()`

**Flow**:
1. Update status to 'in_progress'
2. Construct messages with system prompt and search results
3. Create OpenAI streaming request (gpt-5 with reasoning)
4. For each streaming chunk:
   - Append to `partial_text`
   - Increment `sequence_number`
   - Record time to first token (once)
   - Update `partial_response` in DB (every 50 chunks)
5. On completion:
   - Calculate token usage
   - Call `mark_completed()` with final response
6. On error:
   - Call `mark_failed()` with error message

**Key Properties**:
- Runs as `asyncio.create_task()` (not awaited)
- No HTTP request lifecycle dependency
- Can run for minutes without timeout concerns
- Handles all exceptions gracefully

### 5. Streamlit Frontend

**File**: `app.py`

**Function**: `generate_legal_response_via_fastapi()`

**Purpose**: Client-side polling orchestration

**Flow**:
1. Submit query to POST /api/chat
2. Receive `response_id`
3. Enter polling loop:
   - Call GET /api/response/{response_id}
   - Update UI with status message
   - Display partial response if available
   - Use adaptive intervals:
     - `streaming`: 500ms
     - `in_progress`: 1000ms
     - `queued`: 2000ms
   - Break on `completed` or `failed`
4. On completion:
   - Call GET /api/response/{response_id}/final
   - Return final response and metrics

**Status Messages** (based on elapsed time):
- < 5s: "Processing your query..."
- < 30s: "Analyzing... (Xs)"
- < 60s: "Researching legal sources... (Xs)"
- < 180s: "Deep reasoning in progress... (Xs)"
- > 180s: "Processing extended analysis... (Xs)"

## Why This Architecture Prevents Cloudflare Timeouts

### Problem with Old Approach
```
Browser → Streamlit (HTTP held open) → Poll OpenAI repeatedly
          ↑                                    ↓
          └────────── 150-300s total ──────────┘

Cloudflare sees: ONE HTTP request lasting 150-300s → TIMEOUT at 100s
```

### Solution with New Approach
```
Browser → FastAPI POST /api/chat → Response in 200ms
   ↓
   └── Poll GET /api/response/{id} every 0.5-2s
       Each poll: 50-100ms
       Cloudflare sees: Many short requests, each < 1s

Meanwhile (independently):
FastAPI → Spawn background task → Stream from OpenAI (150-300s)
          ↓
          Write to Supabase incrementally
```

**Key Insight**: No single HTTP request exceeds 10s, let alone 100s. The long-running work happens in a background coroutine that's completely decoupled from HTTP.

## Deployment Configuration

### Environment Variables

Required in `.env` or deployment platform:

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key

# OpenAI
OPENAI_API_KEY=your-openai-key

# FastAPI (for Streamlit to connect to backend)
FASTAPI_URL=http://localhost:8000  # Local dev
FASTAPI_URL=https://your-backend.herokuapp.com  # Production

# Ports (optional, have defaults)
FASTAPI_PORT=8000
PORT=8501  # Streamlit port
```

### Local Development

```bash
# Terminal 1: Start FastAPI
uvicorn fastapi_backend:app --reload --port 8000

# Terminal 2: Start Streamlit
streamlit run app.py

# Or use the combined script
bash start_servers.sh
```

### Production Deployment

**Option 1: Single Platform (Heroku)**
```bash
# Procfile runs both servers
web: bash start_servers.sh
```

**Option 2: Separate Services**
- Deploy FastAPI to Cloud Run, ECS, or similar
- Deploy Streamlit to Streamlit Cloud
- Set `FASTAPI_URL` to point to FastAPI service

**Option 3: Docker Compose**
```yaml
services:
  fastapi:
    build: .
    command: uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY
      - SUPABASE_URL
      - SUPABASE_SERVICE_KEY

  streamlit:
    build: .
    command: streamlit run app.py
    ports:
      - "8501:8501"
    environment:
      - FASTAPI_URL=http://fastapi:8000
    depends_on:
      - fastapi
```

## Request Flow Example

### User asks: "What are the data privacy requirements in California?"

**Timeline**:

```
T=0s    : User submits query in Streamlit
T=0.1s  : Streamlit calls POST /api/chat
T=0.3s  : FastAPI creates DB record, spawns background task, returns response_id
T=0.3s  : Streamlit starts polling GET /api/response/{id}
T=0.5s  : Background task updates status to 'in_progress'
T=0.5s  : Poll #1 returns status='in_progress', partial=''
T=1.5s  : Poll #2 returns status='in_progress', partial=''
T=2.5s  : Poll #3 returns status='in_progress', partial=''
T=4.2s  : OpenAI sends first token
T=4.2s  : Background task records time_to_first_token=3.9s
T=4.5s  : Poll #4 returns status='streaming', partial='California has...'
T=5.5s  : Poll #5 returns status='streaming', partial='California has strict data privacy...'
...     : Polling continues every 0.5s
T=180s  : OpenAI streaming completes (3 minutes total)
T=180.1s: Background task calls mark_completed()
T=180.5s: Poll #356 returns status='completed'
T=180.6s: Streamlit calls GET /api/response/{id}/final
T=180.7s: User sees final response with token usage

Total Cloudflare-visible requests:
- 1 POST /api/chat: 200ms
- 356 GET /api/response/{id}: 50-100ms each
- 1 GET /api/response/{id}/final: 100ms

Longest single request: < 500ms ✓ No timeout!
```

## Error Handling

### Background Task Failures

If `process_openai_background()` throws an exception:
1. Caught by try/except
2. Calls `manager.mark_failed(response_id, error_msg)`
3. Sets status='failed', error_message=details
4. Next poll returns status='failed'
5. Streamlit displays error to user

### Network Failures (Polling)

If poll request fails:
1. Streamlit catches `requests.exceptions.RequestException`
2. Logs error
3. Waits 2s and retries
4. User sees "Attempting to reconnect..." status

### Database Failures

If `create_pending_response()` fails:
1. FastAPI returns HTTP 500
2. Streamlit displays error
3. No background task is spawned (safety)

## Performance Characteristics

### Latency
- Submit query: 200-500ms (DB write + task spawn)
- Each poll: 50-100ms (single DB read)
- First visible text: ~4-5s (TTFT)
- Total time: Depends on LLM (30s - 300s)

### Database Load
- Writes: 1 create + ~100-200 updates per response
- Reads: ~200-600 lightweight selects per response
- Indexes ensure O(1) lookups by response_id

### Concurrency
- Each user gets dedicated background task
- Tasks are independent (no shared state)
- Limited only by:
  - OpenAI rate limits
  - Database connection pool
  - Server memory/CPU

## Monitoring and Observability

### Key Metrics to Track

1. **Response Latency**:
   - Time from submit to first token
   - Time from submit to completion
   - Poll frequency

2. **Failure Rates**:
   - Background task errors
   - OpenAI API errors
   - Database errors

3. **Queue Depth**:
   - Responses in 'queued' state (should be 0-1)
   - Responses in 'in_progress' or 'streaming' (active work)
   - Responses in 'completed' vs 'failed'

4. **Cost Tracking**:
   - Token usage per query
   - Estimated costs by reasoning_effort

### Logging

All modules include structured logging:
- `[FASTAPI]` - HTTP API events
- `[BACKGROUND]` - Background processing events
- `[STREAMLIT]` - Frontend events

## Recovery and Edge Cases

### Page Reload During Processing

**Scenario**: User refreshes browser while LLM is processing

**Solution**:
1. Streamlit's `session_id` is regenerated
2. Call `check_and_recover_pending_response(session_id)`
3. If active response exists, resume polling
4. User sees "Recovering your response..."
5. Continue from last known state

### Orphaned Responses

**Scenario**: Background task crashes without marking failed

**Solution**:
1. Run periodic cleanup job
2. Find responses with status='in_progress' and `updated_at` > 10 minutes old
3. Mark as 'failed' with message="Task timed out"

### Database Connection Loss

**Scenario**: Supabase temporarily unavailable

**Solution**:
1. Background task retries DB writes with exponential backoff
2. If retries exhausted, task fails gracefully
3. User sees "Database temporarily unavailable"

## Testing Strategy

### Unit Tests
- BackgroundResponseManager methods
- Status transition logic
- Token calculation

### Integration Tests
- POST /api/chat → verify DB record created
- Background task → verify status updates
- Poll endpoint → verify correct data returned

### End-to-End Tests
- Submit query → poll until complete → verify response
- Submit query → simulate error → verify error handling
- Submit multiple concurrent queries → verify isolation

### Load Tests
- 10 concurrent users
- 100 sequential queries
- Monitor database performance

## Security Considerations

### Authentication
- All endpoints check `access_code` parameter
- Database RLS policies enforce user isolation
- Service role key used only in backend

### Input Validation
- Query length limits
- Reasoning effort validation
- SQL injection prevention (parameterized queries)

### Rate Limiting
- TODO: Add rate limiting to FastAPI endpoints
- Prevent abuse of polling endpoints

## Future Enhancements

1. **WebSocket Support**: Replace polling with WebSocket for true real-time updates
2. **Priority Queue**: High-priority queries processed first
3. **Response Caching**: Cache similar queries
4. **Multi-Model Support**: Easy switching between GPT-5, Claude, etc.
5. **Analytics Dashboard**: Visualize usage, costs, performance
6. **A/B Testing**: Compare different reasoning efforts

## Troubleshooting

### "Connection refused" errors
- Ensure FastAPI is running on correct port
- Check `FASTAPI_URL` environment variable
- Verify firewall/security group settings

### Responses stuck in "queued"
- Check FastAPI logs for background task errors
- Verify OpenAI API key is valid
- Check OpenAI rate limits

### Slow polling
- Check database performance
- Verify indexes are created
- Monitor network latency

### High costs
- Review reasoning_effort usage (prefer 'medium')
- Implement query length limits
- Add usage quotas per user

## Conclusion

This architecture successfully decouples long-running LLM processing from HTTP request lifecycles, ensuring compatibility with Cloudflare's 100s timeout while maintaining excellent user experience through efficient polling.

**Key Takeaway**: The long-running work happens in a background task that writes to a database. The frontend polls the database state via fast HTTP requests. No single HTTP request ever waits for the full LLM response.
