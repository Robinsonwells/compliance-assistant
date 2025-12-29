# Content Filter Handling - Quick Start Guide

## For Developers: 5-Minute Overview

### What Happens When Content Filter Triggers?

```
User Prompt (blocked)
    ↓
Polling detects content_filter
    ↓
Raise ContentFilterError
    ↓
UI calls generate_safe_rewrite(prompt)
    ↓
gpt-4o-mini generates 3 compliant alternatives
    ↓
Show 3 "Use Option X" buttons
    ↓
User clicks button
    ↓
Auto-submit via pending_prompt
    ↓
Process rewritten prompt normally
```

### Key Functions

#### 1. generate_safe_rewrite(original_prompt: str) → List[str]
**What:** Calls gpt-4o-mini to generate 3 compliant rewrites
**When:** Called in UI exception handler
**Returns:** Always 3 strings (has fallbacks)
**Cost:** ~$0.0001 per call

#### 2. ContentFilterError Exception
**Fields:**
- `response_id` - OpenAI response ID
- `elapsed` - Time before block
- `original_prompt` - User's prompt
- `rag_meta` - Dict with input_chars, rag_chunks
- `reason` - "content_filter"

### Code Locations

| What | Where | Lines |
|------|-------|-------|
| Exception class | app.py | 47-55 |
| Rewrite function | app.py | 58-132 |
| Detection logic | app.py | 1040-1064 |
| UI handler | app.py | 1622-1690 |
| Auto-submit | app.py | 1468-1476 |

### Logging Quick Reference

```python
# Content filter detected
[POLICY] response_id=... elapsed=...s reason=content_filter input_chars=... rag_chunks=...

# Rewrite generation
[REWRITE] Calling gpt-4o-mini to generate compliant rephrases...
[REWRITE] SUCCESS: Generated 3 rewrites

# In UI
[POLICY-BLOCK] ContentFilterError caught in UI: response_id=... elapsed=...s
[POLICY-BLOCK] RAG meta: {...}
```

### Testing in 3 Steps

1. **Trigger:** Use prompt that hits content filter
2. **Verify:** See policy block card with 3 options
3. **Click:** Button should auto-submit rewrite

### Common Issues & Fixes

**Issue:** Generic rewrites appearing
- **Cause:** API failure, using fallbacks
- **Fix:** Check OpenAI API key, network

**Issue:** Buttons not working
- **Check:** Button keys unique per response_id
- **Check:** pending_prompt set correctly

**Issue:** Still seeing "timeout" message
- **Check:** ContentFilterError being caught
- **Check:** Not falling through to general exception handler

### Safety Notes

✅ **Safe:** Rewrites are compliance-focused only
✅ **Fallback:** Always returns 3 options (never fails)
✅ **Cost:** Negligible (~$0.0001 per use)
✅ **No bypass:** System prompt enforces compliance

### API Configuration

```python
# In generate_safe_rewrite()
openai_client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=400,
    response_format={"type": "json_object"}
)
```

### UI Components

```python
# Display warning
st.warning("This request couldn't be answered due to content restrictions...")

# Show rewrites
for i, rewrite in enumerate(rewrites, 1):
    with st.expander(f"**Option {i}** (click to expand)"):
        st.markdown(rewrite)

    if st.button(f"✅ Use Option {i}", key=f"rewrite_{response_id}_{i}"):
        st.session_state.pending_prompt = rewrite
        st.rerun()
```

### Auto-Submit Pattern

```python
# In main app (before chat input)
if st.session_state.get('pending_prompt'):
    prompt = st.session_state.pending_prompt
    del st.session_state['pending_prompt']
    handle_chat_input(prompt)
```

### Error Handling

```python
try:
    rewrites = generate_safe_rewrite(prompt)
except Exception as e:
    # Use fallback rewrites
    rewrites = [
        "What are the key compliance considerations...",
        "How do legal organizations approach...",
        "What ethical oversight mechanisms..."
    ]
```

### Verification Checklist

- [ ] Syntax valid: `python3 -m py_compile app.py`
- [ ] ContentFilterError: 5 references
- [ ] generate_safe_rewrite: 3 references
- [ ] pending_prompt: 5 references
- [ ] No new dependencies
- [ ] Backward compatible

### Production Deployment

1. Deploy code
2. Monitor logs for `[POLICY]` and `[REWRITE]` patterns
3. Test with known content-filter prompts
4. Verify auto-submit works end-to-end
5. Check Railway logs for any errors

### Support Resources

- **Full docs:** AI_REWRITE_IMPLEMENTATION.md
- **Summary:** POLICY_BLOCK_REWRITE_SUMMARY.md
- **This guide:** QUICKSTART_CONTENT_FILTER.md

---

**Questions?** Check logs for `[POLICY-BLOCK]` and `[REWRITE]` lines
**Issues?** Verify OpenAI API key and network connectivity
**Testing?** Use known content-filter prompts and verify button behavior
