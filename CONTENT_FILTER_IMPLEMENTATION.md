# Content Filter Handling Implementation

## Overview
Implemented explicit, user-visible handling for OpenAI content filtering to clearly distinguish policy blocks from network/timeout errors.

## Changes Made

### 1. ContentFilterError Exception Class (Lines 47-55)
- Custom exception type for policy blocks
- Fields: `reason`, `response_id`, `elapsed`, `suggested_prompts`, `partial_len`
- Raised when `status="incomplete"` with `incomplete_details.reason == "content_filter"`

### 2. Safe Rephrase Function (Lines 58-117)
**Function:** `suggest_safe_rephrases(original: str) -> List[str]`
- Deterministic, rule-based rewriting (no API calls)
- Generates 3 safe alternatives
- Reframes away from litigation/strategy/maximizing language
- Redirects toward governance/compliance/risk-management/ethics language

**Rephrase Strategy:**
1. **Option 1:** Risk management and compliance considerations
2. **Option 2:** Capacity planning and budget allocation
3. **Option 3:** Ethical oversight and quality control processes

### 3. Polling Loop Updates (Lines 1005-1051)
**Location:** `generate_legal_response_polling()`

**Incomplete Status Handling:**
- Extract `incomplete_details.reason` from response
- Calculate `partial_len` safely (only for strings)
- Log concise diagnostic line once: `[INCOMPLETE] response_id=... elapsed=... reason=... partial_len=...`

**Content Filter Detection:**
- If `reason == "content_filter"`:
  - Log: `[POLICY] response_id=... elapsed=... reason=content_filter input_chars=... rag_chunks=... partial_len=...`
  - Generate 3 safe rephrases using `suggest_safe_rephrases(query)`
  - Mark as failed in background manager
  - Raise `ContentFilterError` (stops polling immediately)
  - **NO fallback generation**

**Other Incomplete Reasons:**
- Log and raise `TimeoutError` (stops polling)
- Include reason and partial_len in error message

### 4. Unknown Status Handling (Lines 1056-1061)
**Changed:** Previously continued polling indefinitely on unknown statuses

**Now:** Raises exception immediately to prevent infinite loops
- Logs: `[POLLING] UNKNOWN STATUS ERROR: ...`
- Marks as failed in background manager
- Raises informative exception

### 5. UI Exception Handling (Lines 1597-1650)
**Location:** `handle_chat_input()` GPT-5 branch

**ContentFilterError Handler:**
1. Clear status/elapsed placeholders
2. Display error message: "⚠️ Response Blocked by Content Policy"
3. Show warning box explaining it's NOT a timeout
4. Display response_id and elapsed time
5. Show section: "💡 Suggested Safe Rephrases"
6. Display 3 clickable buttons with suggested prompts
7. Append policy block message to chat history
8. Set tokens/cost to 0
9. **DO NOT** call `display_streaming_error()`
10. **DO NOT** attempt fallback generation

**Button Click Behavior:**
- Each suggestion shown as button with preview (first 100 chars)
- Clicking sets `st.session_state.selected_suggestion`
- Triggers `st.rerun()` to reprocess

### 6. Selected Suggestion Handling (Lines 1449-1453)
**Location:** Main app flow before `st.chat_input()`

**Logic:**
- Check `st.session_state.get('selected_suggestion')`
- If set:
  - Extract prompt from selected_suggestion
  - Clear session state variable
  - Call `handle_chat_input(prompt)` with new prompt
- Otherwise: proceed with normal chat input

## Logging Strategy

### Content Filter Block
```
[POLICY] response_id=resp_abc123 elapsed=45.2s reason=content_filter input_chars=350 rag_chunks=5 partial_len=0
[POLICY-BLOCK] ContentFilterError caught in UI: response_id=resp_abc123 elapsed=45.2s
```

### Other Incomplete Reasons
```
[INCOMPLETE] response_id=resp_abc123 elapsed=120.5s status=incomplete reason=max_tokens partial_len=5000
[INCOMPLETE] TERMINAL: OpenAI returned status=incomplete. reason=max_tokens partial_len=5000
```

### Unknown Status
```
[POLLING] UNKNOWN STATUS ERROR: Unknown response status: expired (not completed/failed/incomplete/queued/in_progress)
```

## User Experience

### Before (Content Filter)
- Infinite polling or misleading timeout error
- "Attempting fallback to non-streaming mode..."
- No clear indication it's a policy block
- No suggestions for how to rephrase

### After (Content Filter)
- Immediate stop on content_filter detection
- Clear UI card: "Response Blocked by Content Policy"
- Explicit message: "This is NOT a timeout"
- 3 clickable safe rephrase suggestions
- Response ID and elapsed time for debugging
- No confusing fallback attempts

### Before (Unknown Status)
- Continued polling indefinitely
- Silent failures or eventual timeout

### After (Unknown Status)
- Immediate exception with clear error message
- No infinite loops
- Clear logging for debugging

## Testing Recommendations

### Test Case 1: Content Filter Trigger
**Prompt:** Something that triggers OpenAI's content filter (e.g., requests for litigation strategy optimization)

**Expected Behavior:**
1. Polling starts normally
2. Retrieve returns `status="incomplete"`, `reason="content_filter"`
3. Polling stops immediately (no further retrieves)
4. UI shows policy block card
5. Three suggestion buttons appear
6. Clicking suggestion rephrases the query
7. Railway logs show: `[POLICY]` line with response_id, elapsed, reason, input_chars, rag_chunks

### Test Case 2: Normal Query
**Prompt:** "What are the GDPR requirements for data processing?"

**Expected Behavior:**
1. Normal polling behavior (no changes)
2. Completes successfully
3. Response displayed
4. No policy block cards

### Test Case 3: Click Suggestion
**Setup:** Trigger content filter, see suggestions

**Action:** Click "Option 2" button

**Expected Behavior:**
1. Page reruns
2. New user message with selected suggestion appears
3. Normal processing begins with rephrased prompt
4. Should complete successfully (if rephrase is valid)

## Edge Cases Handled

1. **Partial text from incomplete:** Only logs length if text is string
2. **Missing incomplete_details:** Safely extracts with `getattr(..., None)`
3. **Empty suggestions list:** Always returns exactly 3 suggestions
4. **Multiple content filter errors:** Each gets unique button keys using response_id
5. **Stale selected_suggestion:** Cleared after use to prevent reprocessing

## Files Modified

- `app.py` (Lines 47-117, 1005-1061, 1449-1453, 1597-1650)

## Dependencies

- No new dependencies added
- Uses existing OpenAI SDK structures
- Uses existing Streamlit UI components

## Backward Compatibility

- Normal queries: unchanged behavior
- Timeouts: unchanged behavior (still shows streaming error + fallback)
- Only content_filter cases have new behavior
- Existing error handling paths preserved
