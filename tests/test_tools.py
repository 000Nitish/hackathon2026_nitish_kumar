from __future__ import annotations

import asyncio
import copy

import pytest
from tenacity import wait_none

from src.schemas import EligibilityResult, EscalateResult, KBResult, OrderResult, RefundResult, ToolError
from src.tools import (
    MAX_RETRY_ATTEMPTS,
    PolicyViolationError,
    check_refund_eligibility,
    escalate,
    get_order,
    get_runtime_sets,
    issue_refund,
    search_knowledge_base,
    send_reply,
)
from src import tools


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch: pytest.MonkeyPatch):
    datastore = tools.get_datastore()
    original_orders = copy.deepcopy(datastore.orders)
    tools.reset_runtime_state()

    monkeypatch.setenv("DETERMINISTIC_FAILURES", "true")
    monkeypatch.setattr(
        tools.get_failure_injector(),
        "should_fail",
        lambda *args, **kwargs: False,
    )

    tools._fetch_order_raw.retry.wait = wait_none()
    tools._fetch_customer_raw.retry.wait = wait_none()
    tools._check_refund_eligibility_with_retry.retry.wait = wait_none()

    yield

    datastore.orders.clear()
    datastore.orders.update(original_orders)
    tools.reset_runtime_state()


@pytest.mark.asyncio
async def test_get_order_returns_valid_schema() -> None:
    result = await get_order("ORD-1001", correlation_id="test-001")
    assert isinstance(result, OrderResult)
    assert result.order_id == "ORD-1001"
    assert result.product_id


@pytest.mark.asyncio
async def test_get_order_handles_timeout_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    async def immediate_timeout(tool_name: str, seed_key: str = "") -> None:
        raise asyncio.TimeoutError(f"Forced timeout in {tool_name} ({seed_key})")

    def forced_fail(tool_name: str, failure_type: str, seed_key: str = "") -> bool:
        return tool_name == "get_order" and failure_type == "timeout"

    monkeypatch.setattr(tools, "_inject_timeout", immediate_timeout)
    monkeypatch.setattr(tools.get_failure_injector(), "should_fail", forced_fail)

    result = await get_order("ORD-1001", correlation_id="test-timeout")
    assert isinstance(result, ToolError)
    assert result.error_type == "timeout"
    assert result.retries_attempted == MAX_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_get_order_returns_tooleror_for_unknown_id() -> None:
    result = await get_order("ORD-9999", correlation_id="test-unknown")
    assert isinstance(result, ToolError)
    assert result.error_type == "not_found"


@pytest.mark.asyncio
async def test_check_refund_eligibility_sets_eligibility_flag() -> None:
    result = await check_refund_eligibility("ORD-1001", correlation_id="eligibility-1")
    checked, _ = get_runtime_sets()

    assert isinstance(result, EligibilityResult)
    assert "ORD-1001" in checked


@pytest.mark.asyncio
async def test_issue_refund_raises_policy_violation_without_eligibility_check() -> None:
    with pytest.raises(PolicyViolationError):
        await issue_refund("ORD-1001", 129.99, correlation_id="refund-policy")


@pytest.mark.asyncio
async def test_issue_refund_prevents_double_refund() -> None:
    eligibility = await check_refund_eligibility("ORD-1001", correlation_id="eligibility-2")
    assert isinstance(eligibility, EligibilityResult)

    first = await issue_refund("ORD-1001", 129.99, correlation_id="refund-1")
    assert isinstance(first, RefundResult)

    second = await issue_refund("ORD-1001", 129.99, correlation_id="refund-2")
    assert isinstance(second, ToolError)
    assert second.error_type == "policy_violation"


@pytest.mark.asyncio
async def test_search_knowledge_base_never_returns_empty() -> None:
    result = await search_knowledge_base("zzqv no matching tokens", correlation_id="kb-1")
    assert isinstance(result, KBResult)
    assert result.results.strip()


@pytest.mark.asyncio
async def test_escalate_accepts_valid_priorities_only() -> None:
    summary = (
        "Customer reported an issue for ORD-1001 and we verified delivery status "
        "after calling verification tools."
    )

    for priority in ("low", "medium", "high", "urgent"):
        result = await escalate("TKT-001", summary, priority, correlation_id="esc-valid")
        assert isinstance(result, EscalateResult)
        assert result.priority == priority

    fallback = await escalate("TKT-001", summary, "critical", correlation_id="esc-invalid")
    assert isinstance(fallback, EscalateResult)
    assert fallback.priority == "medium"


@pytest.mark.asyncio
async def test_send_reply_requires_customer_name() -> None:
    bad_message = (
        "We reviewed your refund request and can move forward once you confirm details."
    )
    bad = await send_reply("TKT-001", bad_message, correlation_id="reply-1")

    assert isinstance(bad, ToolError)
    assert bad.error_type == "malformed"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_transient(tool_name: str, failure_type: str, seed_key: str = "") -> bool:
        return tool_name == "check_refund_eligibility" and failure_type == "transient"

    monkeypatch.setattr(tools.get_failure_injector(), "should_fail", always_transient)

    circuit_open_seen = False
    for _ in range(6):
        result = await check_refund_eligibility("ORD-1001", correlation_id="cb-test")
        if isinstance(result, ToolError) and result.error_type == "circuit_open":
            circuit_open_seen = True
            break

    assert circuit_open_seen
