# Content Filter Handling - Quick Reference

## For Developers

### What Changed?

**Before:** Content filter blocks looked like timeouts or network errors

**After:** Content filter blocks have dedicated UI with safe rephrase suggestions

### New Exception Type

```python
class ContentFilterError(Exception):
    def __init__(self, reason, response_id, elapsed, suggested_prompts, partial_len=0):
        ...
```

**When raised:** `status="incomplete"` AND `incomplete_details.reason == "content_filter"`

**What happens:**
- Polling stops immediately
- UI shows policy block card
- 3 clickable rephrase buttons appear
- NO fallback generation attempted

### Key Functions

#### `suggest_safe_rephrases(original: str) -> List[str]`
- **Purpose:** Generate 3 safe alternative prompts
- **Strategy:** Reframe from litigation/strategy/maximizing → governance/compliance/ethics
- **Runtime:** Deterministic, rule-based (no API calls)
- **Always returns:** Exactly 3 strings

#### Updated: `generate_legal_response_polling()`
- **New behavior:** Detects `content_filter` reason in incomplete status
- **New logging:** `[POLICY]` line with response_id, elapsed, reason, input_chars, rag_chunks
- **New exception:** Raises `ContentFilterError` with suggested prompts
- **Bug fix:** Unknown statuses now raise exception (prevents infinite loops)

#### Updated: `handle_chat_input()`
- **New exception handler:** `except ContentFilterError as e:`
- **UI display:** Error card + warning + 3 suggestion buttons
- **No fallback:** Skips `display_streaming_error()` and retry logic
- **Chat history:** Adds policy block message to conversation

### Logging Quick Reference

```python
# Content filter detected
[POLICY] response_id=... elapsed=...s reason=content_filter input_chars=... rag_chunks=... partial_len=...

# Content filter caught in UI
[POLICY-BLOCK] ContentFilterError caught in UI: response_id=... elapsed=...s

# Other incomplete reasons
[INCOMPLETE] response_id=... elapsed=...s status=incomplete reason=... partial_len=...
[INCOMPLETE] TERMINAL: OpenAI returned status=incomplete. reason=... partial_len=...

# Unknown status
[POLLING] UNKNOWN STATUS ERROR: Unknown response status: ... (not completed/failed/incomplete/queued/in_progress)
```

### Testing Checklist

- [ ] Normal queries still work (no regression)
- [ ] Content filter triggers show policy block card
- [ ] 3 suggestion buttons appear on content filter
- [ ] Clicking suggestion button rephrases query
- [ ] Rephrased query processes normally
- [ ] Railway logs show `[POLICY]` line on content filter
- [ ] No infinite loops on unknown statuses
- [ ] Partial text length only logged for strings

### Common Issues

**Issue:** Buttons don't work
- **Check:** Button keys use unique response_id: `key=f"suggest_{e.response_id}_{i}"`
- **Check:** Session state variable cleared after use

**Issue:** Infinite polling
- **Check:** All status branches end with exception or return
- **Check:** Unknown statuses raise exception

**Issue:** len() called on non-string
- **Check:** Use `partial_len = len(partial) if isinstance(partial, str) else 0`

**Issue:** Suggestions not showing
- **Check:** `e.suggested_prompts` has 3 items
- **Check:** `enumerate(e.suggested_prompts, 1)` starts at 1

## For Users

### What to Expect

#### Normal Query
1. Submit question
2. See "Processing..." status
3. Response appears
4. Continue conversation

#### Content Filter Trigger
1. Submit question
2. See "Processing..." status
3. See **"Response Blocked by Content Policy"** message
4. See 3 suggested rephrase buttons
5. Click one to try a safer version
6. Rephrased query should work

### Example Policy Block UI

```
⚠️ Response Blocked by Content Policy

This is NOT a timeout. OpenAI's content filter blocked this
response due to policy restrictions (reason: content_filter).

Response ID: resp_abc123
Elapsed: 45.2s

💡 Suggested Safe Rephrases

These alternatives preserve the complexity of your question
but reframe it using governance, compliance, and risk-management
language:

[Button] Option 1: What are the key compliance considerations
         and risk management factors when evaluating legal
         disputes, including regulatory requirements...

[Button] Option 2: How do law firms approach capacity planning
         and budget allocation for legal matters, including
         assessment criteria for resource deployment...

[Button] Option 3: What ethical oversight mechanisms and quality
         control processes guide legal decision-making,
         including peer review systems, compliance audits...
```

### When to Contact Support

- If normal queries are blocked (false positives)
- If all 3 suggestions also get blocked
- If buttons don't respond to clicks
- If same rephrase gets blocked repeatedly

### Tips for Avoiding Content Filter

**Instead of asking about:**
- Litigation strategy
- Case selection
- Maximizing outcomes
- Winning tactics
- Funding optimization

**Ask about:**
- Compliance requirements
- Risk management factors
- Governance frameworks
- Ethical guidelines
- Professional responsibility standards
- Capacity planning
- Budget allocation
- Quality controls
- Audit processes

## Architecture Notes

### Why No API Call for Rephrases?

**Decision:** Use deterministic rule-based rewriting

**Reasoning:**
1. **Speed:** Instant (no API latency)
2. **Reliability:** No API failures possible
3. **Consistency:** Same input = same output
4. **Cost:** Zero cost
5. **Simplicity:** No new dependencies

**Trade-off:** Less contextual than AI-generated rephrases

**Mitigation:** Provide 3 different angles (compliance, budgeting, ethics)

### Why Stop Polling Immediately?

**Decision:** Raise exception on `content_filter` detection

**Reasoning:**
1. **User experience:** Don't waste user's time
2. **Resource efficiency:** Stop consuming API calls
3. **Clear signal:** Exception = terminal state
4. **Logging clarity:** Single clear failure reason

**Alternative considered:** Continue polling to see if status changes
**Rejected because:** `content_filter` is terminal per OpenAI docs

### Why No Fallback Generation?

**Decision:** Skip retry logic for `ContentFilterError`

**Reasoning:**
1. **Not a transient error:** Policy block won't succeed on retry
2. **User clarity:** Don't confuse user with "Attempting fallback..."
3. **Efficiency:** Don't waste time on doomed retries
4. **Accurate messaging:** Make it clear this is policy, not technical

**Result:** User immediately sees policy block message and suggestions

## File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 47-55 | ContentFilterError class |
| `app.py` | 58-117 | suggest_safe_rephrases() |
| `app.py` | 1005-1051 | Incomplete handling in polling |
| `app.py` | 1056-1061 | Unknown status handling |
| `app.py` | 1449-1453 | Selected suggestion processing |
| `app.py` | 1597-1650 | ContentFilterError UI handler |

## Related Documentation

- `CONTENT_FILTER_IMPLEMENTATION.md` - Detailed implementation notes
- `CONTENT_FILTER_FLOW.md` - Visual flow diagrams
- `test_content_filter.py` - Test script for rephrase function

## Version Info

- **Added:** 2025-12-29
- **OpenAI SDK:** Requires Responses API support
- **Python:** 3.8+
- **Streamlit:** Compatible with current version
- **Dependencies:** None added
