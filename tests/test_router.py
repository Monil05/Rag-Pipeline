import pytest

from retrieval.agent import _call_router


def test_router_returns_direct_answer_for_greeting():
    decision = _call_router("hi")

    assert (
        decision["needs_tools"] is False
    ), f"expected needs_tools=False but got {decision['needs_tools']}"

    print("✓ Greeting router test passed")


def test_router_routes_leave_policy_to_company_docs():
    decision = _call_router("what products do you offer?")

    assert (
        decision["needs_tools"] is True
    ), f"expected needs_tools=True but got {decision['needs_tools']}"
    assert (
        decision["company_needed"] is True
    ), f"expected company_needed=True but got {decision['company_needed']}"
    assert (
        decision["history_needed"] is False
    ), f"expected history_needed=False but got {decision['history_needed']}"

    print("✓ Company query router test passed")
