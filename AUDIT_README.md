# Audit Investigation System

## Quick Start

### Problem
The fact-check audit was returning citations about California property tax when asked to verify Nevada/Arizona overtime law. This investigation system detects and prevents such mismatches.

### Solution Overview

✅ **Diagnostic Logging** - Captures everything sent to/from Perplexity API
✅ **Topic Validation** - Automatically detects off-topic citations
✅ **User Warnings** - Shows prominent alerts when citations are irrelevant
✅ **Test Script** - Reproduces and investigates specific issues

## Files

| File | Purpose |
|------|---------|
| `INVESTIGATION_SUMMARY.md` | **Start here** - Executive summary of problem and solution |
| `AUDIT_INVESTIGATION_GUIDE.md` | Detailed documentation and usage instructions |
| `test_audit_mismatch.py` | Test script to reproduce and investigate issues |
| `AUDIT_README.md` | This file - quick navigation guide |

## Quick Actions

### For Developers

**Investigate an issue**:
```bash
# Run the test script
python3 test_audit_mismatch.py

# Check console output for:
# - Key topics extracted
# - Citations received
# - Relevance scores
# - Mismatch warnings
```

**Check logs during app usage**:
```bash
# Run the app and watch terminal output
streamlit run app.py

# Look for diagnostic sections:
# 🔍 AUDIT FUNCTION CALLED
# 📤 SENDING TO PERPLEXITY API
# 📥 RECEIVED FROM PERPLEXITY API
# ⚠️  RELEVANCE CHECK
```

### For Users

**Look for warnings** in audit reports:

If you see this box, the audit citations may be unreliable:
```
⚠️ CITATION QUALITY WARNING

The fact-checker returned sources that may not be relevant...
```

**What to do**:
- Don't blindly trust the audit result
- Verify independently using official sources
- Report the issue to developers

## How It Works

1. **Extract Topics**: System identifies key topics from your question (states, legal terms)
2. **Send Audit Request**: Sends question and answer to Perplexity for fact-checking
3. **Receive Citations**: Gets back search results and citations
4. **Validate Relevance**: Compares query topics vs citation topics
5. **Calculate Score**: Match score = % of query keywords found in citations
6. **Classify Result**:
   - **GOOD_MATCH** (30%+): ✅ Citations are relevant
   - **WEAK_MATCH** (10-30%): ⚠️ Borderline relevance
   - **CRITICAL_MISMATCH** (<10%): 🚨 Citations are off-topic
7. **Warn User**: If mismatch detected, show prominent warning

## Example

**Query**: "Nevada and Arizona overtime rules"

**Good Audit**:
- Topics found: nevada, arizona, overtime, labor
- Match score: 60%
- Status: ✅ GOOD_MATCH

**Bad Audit**:
- Topics found: california, property, tax
- Match score: 0%
- Status: 🚨 CRITICAL_MISMATCH
- Warning displayed to user

## Diagnostic Output Sections

When running the app or test script, you'll see these sections in the console:

```
================================================================================
🔍 AUDIT FUNCTION CALLED
================================================================================
Shows: What query and answer are being verified

🔑 Key topics extracted from query: [list of expected topics]

================================================================================
📤 SENDING TO PERPLEXITY API
================================================================================
Shows: Prompt details, filters, model settings

================================================================================
📥 RECEIVED FROM PERPLEXITY API
================================================================================
Shows: Response status, number of citations

📚 SEARCH RESULTS RECEIVED:
Shows: Each citation with title, URL, snippet

⚠️  RELEVANCE CHECK: [GOOD_MATCH | WEAK_MATCH | CRITICAL_MISMATCH]
Shows: Match score, matched keywords, citation topics

================================================================================
✅ AUDIT COMPLETE
================================================================================
Shows: Final stats, warnings if any
```

## Read Next

1. **INVESTIGATION_SUMMARY.md** - Full explanation of the problem and solution
2. **AUDIT_INVESTIGATION_GUIDE.md** - Detailed usage instructions and diagnostics
3. **test_audit_mismatch.py** - Review the test code to understand validation logic

## Support

If you encounter audit mismatches:

1. Run `python3 test_audit_mismatch.py` with the specific query
2. Check console output for diagnostic information
3. Review the relevance check section
4. Document the expected vs received topics
5. Consider adjusting search parameters if needed

---

**Implementation Date**: 2025-11-06
**Status**: ✅ Active in production
