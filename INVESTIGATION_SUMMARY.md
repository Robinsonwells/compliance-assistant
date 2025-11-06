# Investigation Summary: Audit Citation Mismatch

## Problem Statement

The Independent Fact-Check audit system returned citations about **California Proposition 19 (property tax law)** when verifying an answer about **Nevada/Arizona overtime employment law**. This represents a complete topic mismatch that makes the audit unreliable and potentially misleading to users.

## Investigation Results

### Root Cause Analysis

The exact cause of the original mismatch cannot be determined retrospectively, but likely causes include:

1. **Perplexity API Search Failure**: The web search component failed to find relevant sources and returned unrelated results
2. **Insufficient Search Context**: The audit prompt may not have provided enough specific context to guide the search effectively
3. **Search Filter Limitations**: Domain filters (*.gov, *.edu) and recency filters ("month") may have excluded relevant sources
4. **API Behavior Variation**: Perplexity's search behavior may be non-deterministic or affected by rate limiting

### Key Findings

- The audit prompt correctly included the original query and answer
- The search domain filters were appropriately restrictive (gov, edu sites only)
- No evidence of variable corruption or code bugs in the audit flow
- The issue appears to be with the **external API search results** rather than the application logic

## Implemented Solutions

### 1. Comprehensive Diagnostic Logging ✅

**What**: Added extensive logging throughout the `call_perplexity_auditor()` function

**Why**: To capture exactly what's being sent to and received from the Perplexity API

**Where**: `app.py` lines 384-650

**Output Includes**:
- Original query and answer (first 200 chars)
- Extracted key topics from query
- Full prompt details (length, first 500 chars)
- API request parameters (model, filters, domains)
- API response status and search result count
- Each citation received (title, URL, snippet)
- Relevance check results and match scores
- Warning flags for mismatches

### 2. Topic Extraction and Relevance Validation ✅

**What**: Created `extract_key_topics()` and `validate_citation_relevance()` functions

**Why**: To automatically detect when citations are off-topic

**Where**: `app.py` lines 301-379

**How It Works**:
- Extracts state names and legal keywords from the original query
- Compares query keywords against citation text (titles, URLs, snippets)
- Calculates a match score (0.0-1.0) showing percentage of keywords found
- Classifies as GOOD_MATCH (30%+), WEAK_MATCH (10-30%), or CRITICAL_MISMATCH (<10%)

**Nevada/Arizona Example**:
- Expected topics: `['nevada', 'arizona', 'overtime', 'hours', 'wage', 'employment']`
- California Prop 19 topics: `['property', 'tax', 'proposition', 'california', 'transfer']`
- Match score: **0%** → CRITICAL_MISMATCH

### 3. User-Facing Warning System ✅

**What**: Added prominent warning boxes when citations are off-topic

**Why**: To prevent users from being misled by irrelevant audit results

**Where**: `app.py` lines 938-964

**Display**:
```
⚠️ CITATION QUALITY WARNING

The fact-checker returned sources that may not be relevant to your question:
- Expected topics: nevada, arizona, overtime, hours, wage
- Received topics: property, tax, proposition, california
- Match score: 0%

This suggests the web search may have failed to find appropriate sources.

Recommendation: Verify the information using official government sources directly.
```

### 4. Standalone Test Script ✅

**What**: Created `test_audit_mismatch.py` for independent testing

**Why**: To allow investigation and reproduction of issues without running the full app

**Usage**:
```bash
python3 test_audit_mismatch.py
```

**Features**:
- Tests the specific Nevada/Arizona overtime scenario
- Tests a simple single-state query for comparison
- Shows all diagnostic output in one place
- Displays relevance scores and warnings
- Interactive (press ENTER to step through tests)

## How This Prevents Future Issues

### Before This Implementation

❌ User receives California property tax citations for Nevada overtime question
❌ No indication that anything is wrong
❌ User trusts the audit and may be misled
❌ No way to investigate what went wrong
❌ No data to diagnose the root cause

### After This Implementation

✅ System detects topic mismatch (0% match score)
✅ **Prominent warning** displayed to user
✅ User knows not to trust the audit result
✅ **Full diagnostic logs** captured in console
✅ Developers can investigate using test script
✅ Root cause data available for analysis

## Testing & Validation

### Manual Testing Steps

1. **Run the test script**:
   ```bash
   python3 test_audit_mismatch.py
   ```

2. **Check console output** for:
   - Key topics extracted: Should show relevant legal terms
   - Search results: Check if titles match the query topic
   - Relevance check: Should show match score
   - Warning detection: Should flag if score is low

3. **Run the app** and ask the Nevada/Arizona question:
   ```bash
   streamlit run app.py
   ```

4. **Look for warnings** in the audit report
   - If citations are off-topic, warning box should appear
   - Expected topics vs received topics should be shown
   - Match score should be displayed

### Expected Behavior

**Good Case** (California minimum wage):
- Keywords: california, minimum, wage, employment
- Citations: CA labor law sites, Department of Labor
- Match score: 40-80%
- Status: GOOD_MATCH
- Warning: None

**Problem Case** (if mismatch occurs):
- Keywords: nevada, arizona, overtime
- Citations: Property tax, unrelated topics
- Match score: 0-5%
- Status: CRITICAL_MISMATCH
- Warning: ⚠️ Displayed prominently

## Files Modified

1. **app.py**:
   - Added `extract_key_topics()` function
   - Added `validate_citation_relevance()` function
   - Enhanced `call_perplexity_auditor()` with comprehensive logging
   - Modified `handle_chat_input()` to display relevance warnings
   - Added topic validation before returning audit results

2. **test_audit_mismatch.py** (new):
   - Standalone test script for reproducing issues
   - Tests Nevada/Arizona overtime case
   - Tests simple California query for comparison

3. **AUDIT_INVESTIGATION_GUIDE.md** (new):
   - Comprehensive documentation of the investigation system
   - Explanation of diagnostic output
   - Usage instructions for developers and users
   - Configuration options and thresholds
   - Potential improvements section

4. **INVESTIGATION_SUMMARY.md** (this file):
   - Executive summary of the investigation
   - Quick reference for the problem and solution

## Recommendations

### Immediate Actions

1. ✅ **Use the system** - Diagnostic logging is now always enabled
2. ✅ **Monitor warnings** - Watch for citation quality warnings in production
3. ✅ **Run tests** - Use `test_audit_mismatch.py` to validate behavior

### Future Improvements

1. **Retry Logic**: If CRITICAL_MISMATCH detected, automatically retry with adjusted search parameters
2. **Prompt Enhancement**: Add more explicit instructions about which laws/statutes to verify
3. **Domain Expansion**: Consider adding more legal source domains (e.g., specific state law sites)
4. **Recency Tuning**: Adjust recency filter based on the query (old laws = broader timeframe)
5. **User Feedback**: Add a "Report Audit Issue" button for users to flag problems

### Configuration to Consider

Current Perplexity search settings:
```python
"search_recency_filter": "month",  # May be too restrictive for established laws
"search_domain_filter": ["*.gov", "law.cornell.edu", "*.edu"],
"max_search_results": 8
```

**Potential adjustments**:
- Change recency to "year" for more comprehensive results
- Add specific state labor department domains
- Increase max_search_results to 10 for better coverage

## Conclusion

The investigation system is now in place and will:

1. **Detect** citation mismatches automatically
2. **Alert** users when audit results are unreliable
3. **Capture** diagnostic data for investigation
4. **Enable** testing and reproduction of issues

This won't prevent the Perplexity API from occasionally returning irrelevant results, but it will **detect and disclose** when it happens, preventing users from being misled.

## Quick Reference

| Task | Command |
|------|---------|
| Run test script | `python3 test_audit_mismatch.py` |
| Run app with diagnostics | `streamlit run app.py` (check console) |
| Read full documentation | See `AUDIT_INVESTIGATION_GUIDE.md` |
| Check for warnings | Look for ⚠️ boxes in audit reports |

---

**Status**: ✅ Implementation Complete
**Date**: 2025-11-06
**Impact**: High - Prevents user misinformation from audit failures
