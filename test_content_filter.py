#!/usr/bin/env python3
"""
Test script for content filter handling functionality
"""

def suggest_safe_rephrases(original: str) -> list:
    """
    Generate 3 safe alternative prompts that preserve complexity but reframe
    away from potentially problematic language toward governance/compliance/risk-management language.
    This is a deterministic, rule-based rewrite (no AI calls).
    """
    original_lower = original.lower()

    # Check for various problematic patterns and create targeted rephrases
    has_litigation = any(word in original_lower for word in ['litigation', 'sue', 'lawsuit', 'case', 'litigate'])
    has_strategy = any(word in original_lower for word in ['strategy', 'tactics', 'approach', 'plan'])
    has_maximize = any(word in original_lower for word in ['maximize', 'optimize', 'best', 'winning'])
    has_selection = any(word in original_lower for word in ['select', 'choose', 'pick', 'which'])
    has_funding = any(word in original_lower for word in ['funding', 'finance', 'payment', 'cost'])

    suggestions = []

    # Suggestion 1: Reframe toward risk management and compliance
    if has_litigation or has_strategy:
        suggestions.append(
            "What are the key compliance considerations and risk management factors when evaluating "
            "legal disputes, including regulatory requirements and ethical obligations that govern "
            "attorney decision-making processes?"
        )
    else:
        suggestions.append(
            "What governance frameworks and compliance standards should attorneys follow when "
            "making decisions about resource allocation and case management, with emphasis on "
            "professional responsibility and ethical guidelines?"
        )

    # Suggestion 2: Reframe toward capacity planning and budgeting
    if has_funding or has_selection:
        suggestions.append(
            "How do law firms approach capacity planning and budget allocation for legal matters, "
            "including assessment criteria for resource deployment that align with fiduciary duties "
            "and professional standards?"
        )
    else:
        suggestions.append(
            "What budgeting methodologies and capacity planning frameworks do legal organizations "
            "use to manage caseloads, ensuring compliance with professional responsibility rules "
            "and maintaining quality client service?"
        )

    # Suggestion 3: Reframe toward ethical controls and oversight
    if has_maximize or has_strategy:
        suggestions.append(
            "What ethical oversight mechanisms and quality control processes guide legal decision-making, "
            "including peer review systems, compliance audits, and professional conduct standards that "
            "ensure adherence to bar association rules and client protection regulations?"
        )
    else:
        suggestions.append(
            "How do legal ethics rules and professional responsibility standards inform attorney "
            "decision-making processes, including oversight by bar associations, internal compliance "
            "reviews, and adherence to client protection guidelines?"
        )

    return suggestions


def test_rephrases():
    """Test the rephrase function with various prompts"""

    test_cases = [
        "How can I maximize my litigation success?",
        "Which cases should I select to get the best funding?",
        "What's the best strategy for winning lawsuits?",
        "What are GDPR requirements?",
        "Tell me about contract law"
    ]

    print("=" * 80)
    print("CONTENT FILTER REPHRASE TESTING")
    print("=" * 80)

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {prompt}")
        print('=' * 80)

        suggestions = suggest_safe_rephrases(prompt)

        assert len(suggestions) == 3, f"Expected 3 suggestions, got {len(suggestions)}"

        for j, suggestion in enumerate(suggestions, 1):
            print(f"\nOption {j}:")
            print(f"  {suggestion}")
            assert len(suggestion) > 50, f"Suggestion too short: {len(suggestion)} chars"
            assert isinstance(suggestion, str), f"Suggestion is not a string: {type(suggestion)}"

        print(f"\n✓ Test case {i} passed: 3 valid suggestions generated")

    print(f"\n{'=' * 80}")
    print("✓ ALL TESTS PASSED")
    print('=' * 80)


if __name__ == "__main__":
    test_rephrases()
