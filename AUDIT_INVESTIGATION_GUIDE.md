# Audit Mismatch Investigation Guide

## Overview

This document explains the investigation system implemented to diagnose and prevent citation mismatches in the fact-checking audit system, where search results from Perplexity API may not match the topic of the original legal query.

## The Problem

The issue reported was that the Independent Fact-Check audit returned citations about **California Proposition 19 (property tax law)** when verifying an answer about **Nevada/Arizona overtime employment law**. This represents a complete topic mismatch that renders the audit useless.

### Possible Root Causes

1. **Search Query Construction Issue**: The prompt sent to Perplexity might not contain sufficient context about the original topic
2. **API Search Failure**: Perplexity's search might fail to find relevant results and return unrelated content
3. **Prompt Ambiguity**: The audit prompt might be vague enough that the AI searches for tangential topics
4. **API Rate Limiting**: Rate limiting or errors could cause degraded search quality
5. **Session/Context Pollution**: Variables could be corrupted or stale data could be passed

## Implemented Solutions

### 1. Comprehensive Diagnostic Logging

**Location**: `app.py` - `call_perplexity_auditor()` function

The auditor function now logs detailed information at every step:

```python
# Input validation
print("🔍 AUDIT FUNCTION CALLED")
print(f"📋 Original Query (first 200 chars): {original_query[:200]}...")
print(f"📝 Main Answer (first 200 chars): {main_answer[:200]}...")

# Prompt being sent
print("📤 SENDING TO PERPLEXITY API")
print(f"Prompt length: {len(prompt)} characters")

# API request details
print(f"🌐 Making request to: {url}")
print(f"📦 Model: {payload['model']}")
print(f"🔍 Search filters: recency={...}, domains={...}")

# Response received
print("📥 RECEIVED FROM PERPLEXITY API")
print(f"Response status: {response.status_code}")
print(f"Search results count: {len(search_results)}")

# Each citation
print("📚 SEARCH RESULTS RECEIVED:")
for result in search_results:
    print(f"  [{i}] {title}")
    print(f"      URL: {url}")
    print(f"      Snippet: {snippet[:100]}...")
```

**How to Use**: Check the console/terminal output when running the application. All audit calls will print detailed diagnostics showing exactly what's being sent and received.

### 2. Topic Extraction and Validation

**Location**: `app.py` - `extract_key_topics()` and `validate_citation_relevance()` functions

Before sending the audit request, the system extracts key topics from the original query:

- **State names**: Detects all 50 U.S. state names and abbreviations
- **Legal keywords**: Overtime, wages, breaks, employment law, etc.

After receiving citations, it validates relevance:

```python
def validate_citation_relevance(query_keywords, citations):
    # Extract text from all citations
    # Check how many query keywords appear in citation text
    # Calculate match score (0.0 to 1.0)

    if match_score >= 0.3:
        return "GOOD_MATCH"
    elif match_score >= 0.1:
        return "WEAK_MATCH"
    else:
        return "CRITICAL_MISMATCH"
```

**Thresholds**:
- **GOOD_MATCH**: 30%+ of query keywords found in citations
- **WEAK_MATCH**: 10-30% match (borderline relevance)
- **CRITICAL_MISMATCH**: <10% match (completely off-topic)

### 3. User-Facing Warnings

**Location**: `app.py` - `handle_chat_input()` function

When citation relevance is poor, the system now displays a warning box to the user:

```
⚠️ CITATION QUALITY WARNING

The fact-checker returned sources that may not be relevant to your question:
- Expected topics: nevada, arizona, overtime, hours, wage, employment
- Received topics: property, tax, proposition, california, transfer, basis
- Match score: 5%

This suggests the web search may have failed to find appropriate sources.
The audit report below should be reviewed critically.

Recommendation: Verify the information using official government sources directly.
```

This makes the problem transparent to users and helps them make informed decisions about trusting the audit results.

### 4. Test Script for Reproduction

**Location**: `test_audit_mismatch.py`

A standalone test script that can reproduce the issue and show diagnostic output:

```bash
python test_audit_mismatch.py
```

The script:
- Tests the Nevada/Arizona overtime question that showed the original mismatch
- Tests a simple California minimum wage query for comparison
- Shows all diagnostic output
- Displays relevance scores and warnings
- Allows investigation without running the full Streamlit app

## How to Use This System

### For Developers

1. **Enable Diagnostic Output**:
   - Diagnostic logging is always enabled in the `call_perplexity_auditor()` function
   - Run the app with: `streamlit run app.py`
   - Watch the terminal/console for detailed diagnostic output

2. **Run the Test Script**:
   ```bash
   python test_audit_mismatch.py
   ```
   This will show you exactly what's happening with the audit function.

3. **Check Relevance Scores**:
   - Look for the "RELEVANCE CHECK" section in console output
   - Match score shows percentage of query keywords found in citations
   - Status shows GOOD_MATCH, WEAK_MATCH, or CRITICAL_MISMATCH

4. **Investigate Specific Cases**:
   - Copy any problematic query into the test script
   - Run it to see full diagnostic output
   - Check what topics are extracted vs what citations are returned

### For Users

When you see a **Citation Quality Warning** in the audit report:

1. **Don't blindly trust the audit** - the citations may be irrelevant
2. **Review the expected vs received topics** to understand the mismatch
3. **Verify information independently** using the official sources mentioned in the main answer
4. **Report the issue** if this happens frequently

## Diagnostic Output Interpretation

### Example Good Match

```
🔑 Key topics extracted from query: ['nevada', 'arizona', 'overtime', 'hours', 'employment']

📚 SEARCH RESULTS RECEIVED:
  [1] Nevada Overtime Law - Department of Labor
      URL: https://labor.nv.gov/uploadedFiles/laborqnvgov/content/Employer/Overtime_Requirements.pdf
      Snippet: Nevada overtime laws require employers to pay 1.5 times...

⚠️  RELEVANCE CHECK: GOOD_MATCH
    Match score: 0.60
    Matched keywords: ['nevada', 'overtime', 'hours', 'employment']
    Citation topics: ['nevada', 'overtime', 'labor', 'requirements', 'wage']
```

### Example Critical Mismatch

```
🔑 Key topics extracted from query: ['nevada', 'arizona', 'overtime', 'hours', 'employment']

📚 SEARCH RESULTS RECEIVED:
  [1] 2025 Property Taxes in California: What You Should Know
      URL: https://www.cnb.com/personal-banking/insights/tax-assessments.html
      Snippet: California Proposition 19 changed property tax reassessment...

⚠️  RELEVANCE CHECK: CRITICAL_MISMATCH
    Match score: 0.00
    Matched keywords: []
    Citation topics: ['property', 'taxes', 'california', 'proposition', 'reassessment']

🚨 CRITICAL: Citations are completely off-topic!
    Expected topics: ['nevada', 'arizona', 'overtime', 'hours', 'employment']
    Received topics: ['property', 'taxes', 'california', 'proposition', 'reassessment']
    This indicates a search failure or API error.
```

## Next Steps for Investigation

If you encounter a mismatch:

1. **Check the console output** to see what prompt was sent to Perplexity
2. **Review the API response** to see what search results were returned
3. **Compare expected vs received topics** to understand the gap
4. **Test with the standalone script** to isolate the issue
5. **Review the Perplexity API request**:
   - Is the prompt clear about what needs to be verified?
   - Are the search domain filters appropriate?
   - Is the search recency filter too restrictive?

## Configuration Options

### Search Parameters (in `call_perplexity_auditor`)

```python
"web_search_options": {
    "search_recency_filter": "month",  # Can be: month, week, day, year
    "search_domain_filter": [
        "*.gov",              # Government sites
        "law.cornell.edu",    # Cornell Law
        "*.edu"               # Educational institutions
    ],
    "max_search_results": 8  # Number of sources to retrieve
}
```

**Considerations**:
- **Recency filter** set to "month" might be too restrictive for established laws
- **Domain filter** is good for legal research but might miss authoritative sources
- **Max results** at 8 is reasonable for performance

### Relevance Thresholds

```python
if match_score >= 0.3:
    status = "GOOD_MATCH"      # 30%+ keywords matched
elif match_score >= 0.1:
    status = "WEAK_MATCH"      # 10-30% keywords matched
else:
    status = "CRITICAL_MISMATCH"  # <10% keywords matched
```

These can be adjusted if needed, but current values seem reasonable.

## Potential Improvements

1. **Retry Logic**: If CRITICAL_MISMATCH is detected, automatically retry with different search parameters
2. **Prompt Enhancement**: Include more explicit context in the audit prompt about which specific laws/statutes to verify
3. **Citation Pre-filtering**: Reject citations that don't contain any query keywords before sending to the AI
4. **Domain Expansion**: Consider adding more authoritative legal sources to the domain filter
5. **Recency Adjustment**: Use different recency filters based on the age of laws being discussed
6. **User Feedback Loop**: Allow users to report false audit results to improve the system

## Conclusion

The implemented system provides comprehensive visibility into the audit process and automatic detection of topic mismatches. By adding diagnostic logging, relevance validation, and user warnings, we can now:

- **Detect** when citations are off-topic
- **Alert** users to potential audit failures
- **Investigate** the root cause using detailed logs
- **Test** specific scenarios independently

This should prevent users from being misled by irrelevant audit results and provide developers with the tools needed to diagnose and fix any underlying API or prompt issues.
