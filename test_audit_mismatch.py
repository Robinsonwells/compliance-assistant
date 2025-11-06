#!/usr/bin/env python3
"""
Test script to investigate audit citation mismatches

This script allows you to test the auditor function independently
and see detailed diagnostic output about what's being sent to
the Perplexity API and what's being returned.

Usage:
    python test_audit_mismatch.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the functions we need to test
from app import call_perplexity_auditor, extract_key_topics, validate_citation_relevance

def test_nevada_arizona_overtime():
    """Test the specific Nevada/Arizona overtime case that showed a mismatch"""

    print("\n" + "="*80)
    print("TEST: Nevada/Arizona Overtime Calculation")
    print("="*80 + "\n")

    # The original question
    original_query = "An employee works split weeks: 30 hours in Nevada and 20 hours in Arizona. How is overtime calculated?"

    # A simplified version of what the main answer would look like
    main_answer = """Short answer: Add all the hours the employee worked for the same employer in the workweek, regardless of the state. Anything over 40 hours in that fixed 7-day workweek is overtime at 1.5× the employee's regular rate. If the Nevada and Arizona work are paid at different rates, compute the week's "regular rate" as a weighted average using all straight-time earnings divided by all hours worked, then pay the 10 overtime hours at 1.5× that regular rate.

Federal weekly overtime (applies in both states): 30 NV hours + 20 AZ hours = 50 hours. That's 10 overtime hours in the week.

Nevada may also require daily overtime for Nevada days if the employee's rate is below 1.5× the Nevada minimum wage.

Legal Basis: Federal FLSA requires overtime after 40 hours per week. Nevada Revised Statutes 608.018 requires overtime for hours over 8 in a day or 40 in a week. Arizona follows federal standards.

This information is for research purposes only and doesn't constitute legal advice. Consult qualified legal counsel for specific guidance."""

    print("📝 Testing with:")
    print(f"   Query: {original_query}")
    print(f"   Answer length: {len(main_answer)} characters\n")

    # Extract topics to show what we expect
    topics = extract_key_topics(original_query)
    print(f"🔑 Expected topics in citations: {topics}\n")

    # Call the auditor
    print("🚀 Calling auditor function...\n")
    print("="*80)

    result = call_perplexity_auditor(original_query, main_answer)

    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)

    print(f"\n✅ Audit completed")
    print(f"   Citations received: {len(result['citations'])}")
    print(f"   Report length: {len(result['report'])} characters")
    print(f"   Tokens used: {result['usage']['total_tokens']}")

    if result.get('relevance_warning'):
        warning = result['relevance_warning']
        print(f"\n⚠️  RELEVANCE WARNING DETECTED:")
        print(f"   Status: {warning['status']}")
        print(f"   Match score: {warning['match_score']:.2%}")
        print(f"   Expected: {warning['query_keywords'][:8]}")
        print(f"   Received: {warning['citation_topics'][:8]}")
    else:
        print(f"\n✅ No relevance warnings - citations appear relevant")

    print("\n" + "="*80)
    print("📚 CITATIONS RECEIVED:")
    print("="*80)
    for i, cite in enumerate(result['citations'], 1):
        print(f"\n[{i}] {cite['title']}")
        print(f"    {cite['url']}")
        if cite.get('snippet'):
            print(f"    Snippet: {cite['snippet'][:150]}...")

    print("\n" + "="*80)
    print("📄 AUDIT REPORT:")
    print("="*80)
    print(result['report'][:1000])
    if len(result['report']) > 1000:
        print(f"\n... (truncated, full report is {len(result['report'])} characters)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    return result

def test_simple_query():
    """Test with a simple, single-state query"""

    print("\n" + "="*80)
    print("TEST: Simple California Minimum Wage Query")
    print("="*80 + "\n")

    original_query = "What is the minimum wage in California?"
    main_answer = """California's minimum wage is $16.00 per hour as of 2024. This applies to all employers regardless of size.

Legal Basis: California Labor Code Section 1182.12 sets the minimum wage. For 2024, it is $16.00 per hour.

This information is for research purposes only and doesn't constitute legal advice. Consult qualified legal counsel for specific guidance."""

    print("📝 Testing with simple query:")
    print(f"   Query: {original_query}")

    topics = extract_key_topics(original_query)
    print(f"🔑 Expected topics: {topics}\n")

    result = call_perplexity_auditor(original_query, main_answer)

    print("\n📊 RESULTS:")
    print(f"   Citations: {len(result['citations'])}")
    print(f"   Has warning: {bool(result.get('relevance_warning'))}")

    if result.get('relevance_warning'):
        print(f"   ⚠️  WARNING: {result['relevance_warning']['status']}")

    return result

def main():
    """Run all tests"""

    # Check for API key
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("❌ ERROR: PERPLEXITY_API_KEY not found in environment")
        print("   Please set it in your .env file")
        sys.exit(1)

    print("\n" + "="*80)
    print("AUDIT MISMATCH INVESTIGATION TEST SUITE")
    print("="*80)
    print("\nThis script will test the auditor function with known queries")
    print("and show detailed diagnostic output to help identify mismatches.")
    print("\nAll diagnostic logs will be printed to console.")
    print("="*80 + "\n")

    input("Press ENTER to start Test 1: Nevada/Arizona Overtime...")
    result1 = test_nevada_arizona_overtime()

    print("\n" + "="*80)
    input("\nPress ENTER to start Test 2: Simple California Query...")
    result2 = test_simple_query()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
    print("\n📋 Summary:")
    print(f"   Test 1 (Multi-state): {len(result1['citations'])} citations, " +
          f"{'WARNING' if result1.get('relevance_warning') else 'OK'}")
    print(f"   Test 2 (Single-state): {len(result2['citations'])} citations, " +
          f"{'WARNING' if result2.get('relevance_warning') else 'OK'}")
    print("\n💡 Check the console output above for detailed diagnostic information.")
    print("   Look for the RELEVANCE CHECK section to see topic matching details.\n")

if __name__ == "__main__":
    main()
