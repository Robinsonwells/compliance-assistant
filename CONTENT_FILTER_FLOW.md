# Content Filter Handling Flow Diagram

## Flow: Normal Query vs Content Filter

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User submits prompt                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│             handle_chat_input() called with prompt                   │
│  - Classify reasoning effort                                         │
│  - Search legal database (RAG)                                       │
│  - Call generate_legal_response_smart()                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│           generate_legal_response_polling() starts                   │
│  - Create background response (stream=False, background=True)        │
│  - Start polling loop                                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                ┌────────────────┴────────────────┐
                │  Poll: openai_client.responses  │
                │        .retrieve(response_id)   │
                └────────────────┬────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │   What is result.status?        │
                └────────────────┬────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  "completed"  │    │   "in_progress"   │    │  "incomplete"    │
└───────┬───────┘    │   "queued"        │    └────────┬─────────┘
        │            └─────────┬─────────┘             │
        │                      │                       │
        │            ┌─────────┴──────────┐            │
        │            │ Sleep 2s           │            │
        │            │ Continue polling   │            │
        │            └────────────────────┘            │
        │                                              │
        ▼                                              ▼
┌─────────────────────────────────┐    ┌──────────────────────────────┐
│ Extract response text           │    │ Extract incomplete_details   │
│ Calculate tokens/cost           │    │ reason = details.reason      │
│ Return success                  │    │ partial_len = len(text)      │
│                                 │    └─────────┬────────────────────┘
│ UI displays response normally   │              │
└─────────────────────────────────┘              │
                                        ┌────────┴────────┐
                                        │ Is reason ==     │
                                        │ "content_filter"?│
                                        └────────┬────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                                    ▼ YES                     ▼ NO
                    ┌───────────────────────────┐   ┌─────────────────┐
                    │ Log [POLICY] line         │   │ Log [INCOMPLETE]│
                    │ Generate 3 safe rephrases │   │ Raise TimeoutErr│
                    │ Raise ContentFilterError  │   └─────────────────┘
                    └──────────┬────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │ Exception caught in handle_chat_   │
              │ input() - ContentFilterError branch│
              └──────────┬─────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UI DISPLAYS POLICY BLOCK                          │
│                                                                      │
│  ⚠️ Response Blocked by Content Policy                              │
│                                                                      │
│  This is NOT a timeout. OpenAI's content filter blocked this        │
│  response due to policy restrictions (reason: content_filter).      │
│                                                                      │
│  Response ID: resp_abc123                                           │
│  Elapsed: 45.2s                                                     │
│                                                                      │
│  💡 Suggested Safe Rephrases                                        │
│                                                                      │
│  These alternatives preserve the complexity of your question        │
│  but reframe it using governance, compliance, and risk-management   │
│  language:                                                          │
│                                                                      │
│  [Button] Option 1: What are the key compliance...                 │
│  [Button] Option 2: How do law firms approach capacity...          │
│  [Button] Option 3: What ethical oversight mechanisms...           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                        User clicks button
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ st.session_state       │
                    │ .selected_suggestion = │
                    │ chosen prompt          │
                    │                        │
                    │ st.rerun()             │
                    └────────┬───────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ Main app checks for        │
                │ selected_suggestion        │
                │ If set: call               │
                │ handle_chat_input(prompt)  │
                └────────┬───────────────────┘
                         │
                         ▼
            ┌────────────────────────────────┐
            │  Process rephrased prompt      │
            │  (should pass content filter)  │
            └────────────────────────────────┘
```

## Key Decision Points

### Status = "incomplete"
```
incomplete_details.reason?
│
├─ "content_filter" → ContentFilterError
│                     → UI shows policy block
│                     → 3 suggestion buttons
│                     → NO fallback
│
├─ "max_tokens"     → TimeoutError
│                     → Standard error handling
│                     → Fallback may be attempted
│
└─ other/null       → TimeoutError
                      → Standard error handling
```

### Unknown Status
```
status not in [completed, failed, incomplete, queued, in_progress]
│
└─ Raise Exception immediately
   → Prevent infinite loop
   → Log [POLLING] UNKNOWN STATUS ERROR
   → Mark as failed in DB
```

## Logging Examples

### Content Filter Block
```
[POLLING] Started background response: resp_abc123
[POLLING] Request params: model=gpt-5, background=True, stream=False, effort=high, verbosity=high
[POLLING] Input size: query=250 chars, context=15000 chars, input_text=16500 chars
[POLLING] RAG data: chunks=5, rag_text_chars=12000

[POLL] Poll #1, elapsed=2.1s, retrieving status...
[POLL] Poll #1, elapsed=2.1s, status=in_progress

[POLL] Poll #2, elapsed=4.3s, retrieving status...
[POLL] Poll #2, elapsed=4.3s, status=in_progress

...

[POLL] Poll #22, elapsed=45.2s, retrieving status...
[POLL] Poll #22, elapsed=45.2s, status=incomplete

[INCOMPLETE] response_id=resp_abc123 elapsed=45.2s status=incomplete reason=content_filter partial_len=0
[POLICY] response_id=resp_abc123 elapsed=45.2s reason=content_filter input_chars=250 rag_chunks=5 partial_len=0

[POLICY-BLOCK] ContentFilterError caught in UI: response_id=resp_abc123 elapsed=45.2s
```

### Normal Completion
```
[POLLING] Started background response: resp_xyz789
[POLLING] Request params: model=gpt-5, background=True, stream=False, effort=medium, verbosity=high
[POLLING] Input size: query=180 chars, context=8500 chars, input_text=10200 chars
[POLLING] RAG data: chunks=5, rag_text_chars=7000

[POLL] Poll #1, elapsed=2.0s, retrieving status...
[POLL] Poll #1, elapsed=2.0s, status=in_progress

...

[POLL] Poll #35, elapsed=71.5s, retrieving status...
[POLL] Poll #35, elapsed=71.5s, status=completed

[COMPLETE] ✅ Finished successfully in 71.5s (1.2 minutes)
[COMPLETE] Response length: 1250 characters
[COMPLETE] Tokens - Input: 3500, Output: 800, Reasoning: 2500, Total: 6800
[COMPLETE] Total polls: 35
```

## Error Prevention

### Infinite Loop Prevention
1. **Hard timeout:** 600s (10 minutes) maximum polling time
2. **Unknown status:** Immediate exception (no continued polling)
3. **Content filter:** Immediate exception (no continued polling)
4. **Other incomplete:** Immediate exception (no continued polling)

### Safe Partial Text Extraction
```python
partial = extract_output_text_from_response(result)
partial_len = len(partial) if isinstance(partial, str) else 0
```
Only call `len()` on verified strings to prevent errors.

### Deterministic Rephrasing
```python
suggestions = suggest_safe_rephrases(query)
```
- No API calls (fast, no failure modes)
- Always returns exactly 3 strings
- Works even for short/empty prompts
- Consistent output for same input

## User Experience Comparison

### BEFORE: Content Filter Triggered
```
Status: Processing your query...
Status: Processing your query...
Status: Processing your query...
[90 seconds pass]

ERROR: TimeoutError
Attempting fallback to non-streaming mode...
[30 more seconds pass]

ERROR: Still failing
Maybe check your internet connection?
```
❌ Confusing, misleading, wastes user time

### AFTER: Content Filter Triggered
```
Status: Processing your query...
Status: Processing your query...
[45 seconds pass]

⚠️ Response Blocked by Content Policy

This is NOT a timeout. OpenAI's content filter blocked this response
due to policy restrictions (reason: content_filter).

Response ID: resp_abc123
Elapsed: 45.2s

💡 Suggested Safe Rephrases

[Option 1 Button] What are the key compliance considerations...
[Option 2 Button] How do law firms approach capacity planning...
[Option 3 Button] What ethical oversight mechanisms...
```
✅ Clear, actionable, saves user time
