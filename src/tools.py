from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from circuitbreaker import CircuitBreakerError, CircuitBreakerMonitor, circuit
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from .schemas import (
    CustomerResult,
    EligibilityResult,
    EscalateResult,
    KBResult,
    OrderResult,
    PriorityType,
    ProductResult,
    RefundResult,
    ReplyResult,
    ToolError,
)


logger = logging.getLogger("shopwave.tools")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "at",
    "with",
    "this",
    "that",
    "i",
    "my",
    "it",
    "be",
    "as",
    "by",
    "from",
}


MAX_RETRY_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
RETRY_WAIT_MIN = int(os.getenv("RETRY_WAIT_MIN", "1"))
RETRY_WAIT_MAX = int(os.getenv("RETRY_WAIT_MAX", "8"))
REFUND_ESCALATION_AMOUNT = float(os.getenv("REFUND_ESCALATION_AMOUNT", "200.00"))


_eligibility_checked_orders: set[str] = set()
_refunds_issued: set[str] = set()
_refund_lock = asyncio.Lock()


class PolicyViolationError(Exception):
    """
    Raised when an irreversible action is attempted without required preconditions.
    This exception should NEVER be caught silently.
    """

    def __init__(self, message: str, tool_name: str, order_id: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.order_id = order_id


class DataStore:
    """
    Loads all JSON files once at startup and provides indexed lookups.
    """

    _instance: "DataStore | None" = None

    def __init__(self) -> None:
        self.customers: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.products: dict[str, dict[str, Any]] = {}
        self.customers_by_id: dict[str, dict[str, Any]] = {}
        self.tickets_by_id: dict[str, dict[str, Any]] = {}
        self.knowledge_base: str = ""
        self.kb_sections: dict[str, str] = {}
        self._load()

    @classmethod
    def get(cls) -> "DataStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        customers_data = _read_json(DATA_DIR / "customers.json")
        orders_data = _read_json(DATA_DIR / "orders.json")
        products_data = _read_json(DATA_DIR / "products.json")
        tickets_data = _read_json(DATA_DIR / "tickets.json")

        self.customers = {item["email"].lower(): dict(item) for item in customers_data}
        self.customers_by_id = {item["customer_id"]: dict(item) for item in customers_data}
        self.orders = {item["order_id"]: dict(item) for item in orders_data}
        self.products = {item["product_id"]: dict(item) for item in products_data}
        self.tickets_by_id = {item["ticket_id"]: dict(item) for item in tickets_data}

        kb_path = DATA_DIR / "knowledge-base.md"
        self.knowledge_base = kb_path.read_text(encoding="utf-8")
        self.kb_sections = _split_kb_sections(self.knowledge_base)

        logger.info(
            "DataStore loaded: %s customers, %s orders, %s products, %s KB sections",
            len(self.customers),
            len(self.orders),
            len(self.products),
            len(self.kb_sections),
        )


class FailureInjector:
    """
    Controls probabilistic failures for realistic mock behavior.
    """

    def __init__(self) -> None:
        self._rates: dict[str, dict[str, float]] = {
            "get_order": {"timeout": 0.0, "malformed": 0.0},
            "get_customer": {"timeout": 0.0, "malformed": 0.0},
            "check_refund_eligibility": {"timeout": 0.0, "transient": 0.0},
            "issue_refund": {},
            "send_reply": {},
            "escalate": {},
            "get_product": {},
            "search_knowledge_base": {},
        }
        self._calls: dict[str, int] = {}

    def should_fail(self, tool_name: str, failure_type: str, seed_key: str = "") -> bool:
        env_key = f"FAIL_RATE_{tool_name.upper()}_{failure_type.upper()}"
        base_rate = self._rates.get(tool_name, {}).get(failure_type, 0.0)
        rate = _safe_float(os.getenv(env_key), base_rate)

        if rate <= 0:
            return False

        deterministic = os.getenv("DETERMINISTIC_FAILURES", "false").lower() == "true"
        call_key = f"{tool_name}:{failure_type}:{seed_key}"
        self._calls[call_key] = self._calls.get(call_key, 0) + 1
        call_no = self._calls[call_key]

        if deterministic:
            seed_material = f"{call_key}:{call_no}".encode("utf-8")
            digest = hashlib.sha256(seed_material).hexdigest()[:16]
            sample = int(digest, 16) / float(16**16)
        else:
            sample = random.random()

        return sample < rate


def _safe_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env value '%s', using default=%s", value, default)
        return default


def _read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}, got {type(data).__name__}")
    return data


def _split_kb_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_header: str | None = None

    for line in markdown.splitlines():
        header_match = re.match(r"^##\s+(.+)$", line.strip())
        if header_match:
            current_header = header_match.group(1).strip()
            sections[current_header] = []
            continue

        if current_header is not None:
            sections[current_header].append(line)

    if not sections:
        return {"Knowledge Base": markdown}

    return {header: "\n".join(content).strip() for header, content in sections.items()}


def log_retry_attempt(retry_state: RetryCallState) -> None:
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "retrying tool call",
        extra={
            "function": retry_state.fn.__name__ if retry_state.fn else "unknown",
            "attempt": retry_state.attempt_number,
            "exception": repr(exception),
        },
    )


def _retry_wait_strategy() -> Any:
    return wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX) + wait_random(0, 0.5)


def _tool_error(
    tool_name: str,
    error_type: str,
    message: str,
    correlation_id: str,
    retries_attempted: int = 0,
) -> ToolError:
    return ToolError(
        tool_name=tool_name,
        error_type=error_type,
        message=message,
        retries_attempted=max(0, retries_attempted),
        correlation_id=correlation_id,
    )


async def _inject_timeout(tool_name: str, seed_key: str = "") -> None:
    delay_seed = hashlib.sha256(f"{tool_name}:{seed_key}".encode("utf-8")).hexdigest()
    deterministic = os.getenv("DETERMINISTIC_FAILURES", "false").lower() == "true"
    if deterministic:
        fraction = int(delay_seed[:8], 16) / float(16**8)
        delay = 0.3 + fraction * 0.9
    else:
        delay = random.uniform(0.3, 1.2)
    await asyncio.sleep(delay)
    raise asyncio.TimeoutError(f"Timeout in {tool_name} after {delay:.2f}s")


class _failure_injector_holder:
    instance: FailureInjector | None = None


def get_datastore() -> DataStore:
    return DataStore.get()


def get_failure_injector() -> FailureInjector:
    if _failure_injector_holder.instance is None:
        _failure_injector_holder.instance = FailureInjector()
    return _failure_injector_holder.instance


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=_retry_wait_strategy(),
    retry=retry_if_exception_type((asyncio.TimeoutError, RuntimeError)),
    before_sleep=log_retry_attempt,
    reraise=False,
)
async def _fetch_order_raw(order_id: str) -> dict[str, Any] | None:
    failure_injector = get_failure_injector()
    datastore = get_datastore()

    if failure_injector.should_fail("get_order", "timeout", seed_key=order_id):
        await _inject_timeout("get_order", seed_key=order_id)

    if order_id.startswith("ORD-9"):
        return None

    order = datastore.orders.get(order_id)
    if order is None:
        return None

    payload = dict(order)
    if failure_injector.should_fail("get_order", "malformed", seed_key=order_id):
        payload.pop("status", None)
        payload.pop("product_id", None)
    return payload


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=_retry_wait_strategy(),
    retry=retry_if_exception_type((asyncio.TimeoutError, RuntimeError)),
    before_sleep=log_retry_attempt,
    reraise=False,
)
async def _fetch_customer_raw(email: str) -> dict[str, Any] | None:
    normalized = email.lower().strip()
    failure_injector = get_failure_injector()
    datastore = get_datastore()

    if failure_injector.should_fail("get_customer", "timeout", seed_key=normalized):
        await _inject_timeout("get_customer", seed_key=normalized)
    return datastore.customers.get(normalized)


@circuit(
    failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
    recovery_timeout=int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30")),
    expected_exception=(RuntimeError, asyncio.TimeoutError),
    name="check_refund_eligibility_circuit",
)
async def _check_refund_eligibility_core(order_id: str) -> EligibilityResult:
    failure_injector = get_failure_injector()
    datastore = get_datastore()

    if failure_injector.should_fail("check_refund_eligibility", "timeout", seed_key=order_id):
        await _inject_timeout("check_refund_eligibility", seed_key=order_id)

    if failure_injector.should_fail("check_refund_eligibility", "transient", seed_key=order_id):
        raise RuntimeError("Service temporarily unavailable: DB connection reset")

    order_raw = datastore.orders.get(order_id)
    if order_raw is None:
        raise LookupError(f"Order not found: {order_id}")

    order = OrderResult.model_validate(order_raw)

    if order.status != "delivered":
        return EligibilityResult(
            order_id=order_id,
            eligible=False,
            reason="Order not yet delivered",
        )

    if (order.refund_status or "").lower() == "refunded":
        refunded_date = order.notes or "unknown date"
        return EligibilityResult(
            order_id=order_id,
            eligible=False,
            reason=f"Already refunded on {refunded_date}",
        )

    product_raw = datastore.products.get(order.product_id)
    if product_raw is None:
        raise LookupError(f"Product not found: {order.product_id}")

    product = ProductResult.model_validate(product_raw)

    if order.delivery_date is None:
        raise RuntimeError("Delivered order missing delivery_date")

    days_since_delivery = (date.today() - order.delivery_date).days
    window = product.return_window_days

    if days_since_delivery > window:
        expired_days = days_since_delivery - window
        return EligibilityResult(
            order_id=order_id,
            eligible=False,
            reason=f"Return window of {window} days expired ({expired_days} days ago)",
            days_since_delivery=days_since_delivery,
            days_remaining=0,
        )

    days_remaining = max(0, window - days_since_delivery)
    if order.amount > REFUND_ESCALATION_AMOUNT:
        return EligibilityResult(
            order_id=order_id,
            eligible=True,
            escalation_required=True,
            reason="Amount exceeds $200 threshold",
            days_since_delivery=days_since_delivery,
            days_remaining=days_remaining,
        )

    return EligibilityResult(
        order_id=order_id,
        eligible=True,
        reason=f"Within return window ({days_remaining} days remaining)",
        days_since_delivery=days_since_delivery,
        days_remaining=days_remaining,
    )


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=_retry_wait_strategy(),
    retry=retry_if_exception_type((asyncio.TimeoutError, RuntimeError)),
    before_sleep=log_retry_attempt,
    reraise=False,
)
async def _check_refund_eligibility_with_retry(order_id: str) -> EligibilityResult:
    return await _check_refund_eligibility_core(order_id)


async def get_order(order_id: str, correlation_id: str = "") -> OrderResult | ToolError:
    """
    Retrieves order details by order_id.
    """
    try:
        raw = await _fetch_order_raw(order_id)
    except RetryError as exc:
        last_exception = exc.last_attempt.exception() if exc.last_attempt else None
        message = "Order service timed out after retries"
        if isinstance(last_exception, RuntimeError):
            message = str(last_exception)
        return _tool_error(
            "get_order",
            "timeout",
            message,
            correlation_id,
            retries_attempted=MAX_RETRY_ATTEMPTS,
        )

    if raw is None:
        return _tool_error(
            "get_order",
            "not_found",
            f"Order '{order_id}' was not found.",
            correlation_id,
        )

    try:
        return OrderResult.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Malformed get_order payload for %s: %s", order_id, exc)
        return _tool_error(
            "get_order",
            "malformed",
            "Order payload failed schema validation.",
            correlation_id,
        )


async def get_customer(email: str, correlation_id: str = "") -> CustomerResult | ToolError:
    """
    Retrieves customer profile by email.
    """
    try:
        raw = await _fetch_customer_raw(email)
    except RetryError as exc:
        last_exception = exc.last_attempt.exception() if exc.last_attempt else None
        message = "Customer service timed out after retries"
        if isinstance(last_exception, RuntimeError):
            message = str(last_exception)
        return _tool_error(
            "get_customer",
            "timeout",
            message,
            correlation_id,
            retries_attempted=MAX_RETRY_ATTEMPTS,
        )

    if raw is None:
        return _tool_error(
            "get_customer",
            "not_found",
            f"Customer '{email}' was not found. Please provide registered email.",
            correlation_id,
        )

    try:
        return CustomerResult.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Malformed get_customer payload for %s: %s", email, exc)
        return _tool_error(
            "get_customer",
            "malformed",
            "Customer payload failed schema validation.",
            correlation_id,
        )


async def get_product(product_id: str, correlation_id: str = "") -> ProductResult | ToolError:
    """
    Retrieves product metadata with no artificial failures.
    """
    raw = get_datastore().products.get(product_id)
    if raw is None:
        return _tool_error(
            "get_product",
            "not_found",
            f"Product '{product_id}' was not found.",
            correlation_id,
        )

    try:
        return ProductResult.model_validate(raw)
    except ValidationError as exc:
        logger.warning("Malformed get_product payload for %s: %s", product_id, exc)
        return _tool_error(
            "get_product",
            "malformed",
            "Product payload failed schema validation.",
            correlation_id,
        )


async def search_knowledge_base(query: str, correlation_id: str = "") -> KBResult | ToolError:
    """
    Simulates semantic search over knowledge-base.md using keyword scoring.
    """
    datastore = get_datastore()

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in STOPWORDS
    ]

    section_scores: list[tuple[str, str, int]] = []
    for header, content in datastore.kb_sections.items():
        text_lower = content.lower()
        header_lower = header.lower()
        score = 0
        for token in tokens:
            score += text_lower.count(token)
            score += 2 * header_lower.count(token)
        section_scores.append((header, content, score))

    if not section_scores:
        return _tool_error(
            "search_knowledge_base",
            "malformed",
            "Knowledge base is not loaded.",
            correlation_id,
        )

    max_score = max(score for _, _, score in section_scores)
    if max_score == 0:
        selected = section_scores
    else:
        selected = sorted(section_scores, key=lambda item: item[2], reverse=True)[:2]

    formatted_sections = [f"## {header}\n{content}" for header, content, _ in selected]
    results_text = "\n---\n".join(part.strip() for part in formatted_sections if part.strip())
    if not results_text.strip():
        results_text = datastore.knowledge_base.strip()

    return KBResult(
        query=query,
        results=results_text,
        sections_searched=len(datastore.kb_sections),
    )


async def check_refund_eligibility(order_id: str, correlation_id: str = "") -> EligibilityResult | ToolError:
    """
    Evaluates refund eligibility with retry and circuit breaker protections.
    """
    try:
        result = await _check_refund_eligibility_with_retry(order_id)
    except CircuitBreakerError:
        return _tool_error(
            "check_refund_eligibility",
            "circuit_open",
            "Service temporarily unavailable — circuit open. Escalate this ticket.",
            correlation_id,
            retries_attempted=MAX_RETRY_ATTEMPTS,
        )
    except RetryError as exc:
        last_exception = exc.last_attempt.exception() if exc.last_attempt else None
        if isinstance(last_exception, LookupError):
            return _tool_error(
                "check_refund_eligibility",
                "not_found",
                str(last_exception),
                correlation_id,
                retries_attempted=0,
            )
        if isinstance(last_exception, asyncio.TimeoutError):
            message = "Eligibility service timed out after retries"
        elif isinstance(last_exception, RuntimeError):
            message = str(last_exception)
        else:
            message = "Eligibility check failed after retries"
        return _tool_error(
            "check_refund_eligibility",
            "timeout",
            message,
            correlation_id,
            retries_attempted=MAX_RETRY_ATTEMPTS,
        )
    except LookupError as exc:
        return _tool_error(
            "check_refund_eligibility",
            "not_found",
            str(exc),
            correlation_id,
            retries_attempted=0,
        )

    _eligibility_checked_orders.add(order_id)
    return result


async def issue_refund(order_id: str, amount: float, correlation_id: str = "") -> RefundResult | ToolError:
    """
    Issues a refund with strict policy guards.
    """
    if order_id not in _eligibility_checked_orders:
        raise PolicyViolationError(
            (
                f"issue_refund called for {order_id} without prior "
                "check_refund_eligibility call. This is a policy violation. "
                "The refund has been BLOCKED."
            ),
            tool_name="issue_refund",
            order_id=order_id,
        )

    datastore = get_datastore()
    order_raw = datastore.orders.get(order_id)
    if order_raw is None:
        return _tool_error(
            "issue_refund",
            "not_found",
            f"Order '{order_id}' not found.",
            correlation_id,
        )

    order = OrderResult.model_validate(order_raw)

    if amount <= 0:
        raise ValueError("Refund amount must be greater than 0")

    max_allowed = order.amount * 1.01
    if amount > max_allowed:
        raise ValueError(
            f"Refund amount {amount:.2f} exceeds maximum allowed {max_allowed:.2f}"
        )

    async with _refund_lock:
        if order_id in _refunds_issued or (order.refund_status or "").lower() == "refunded":
            return _tool_error(
                "issue_refund",
                "policy_violation",
                f"Refund already issued for order '{order_id}' in this session.",
                correlation_id,
            )

        datastore.orders[order_id]["refund_status"] = "refunded"
        datastore.orders[order_id]["notes"] = (
            f"Refunded at {datetime.now(timezone.utc).isoformat()}"
        )
        _refunds_issued.add(order_id)

    refund = RefundResult(
        order_id=order_id,
        transaction_id=str(uuid.uuid4()),
        amount=amount,
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        "[REFUND ISSUED] [%s] %s | $%.2f",
        correlation_id,
        order_id,
        amount,
    )
    return refund


async def send_reply(ticket_id: str, message: str, correlation_id: str = "") -> ReplyResult | ToolError:
    """
    Sends the final response to the customer.
    """
    clean_message = message.strip()
    if len(clean_message) <= 20:
        return _tool_error(
            "send_reply",
            "malformed",
            "Reply message must be more than 20 characters.",
            correlation_id,
        )

    datastore = get_datastore()
    ticket = datastore.tickets_by_id.get(ticket_id)
    if ticket is None:
        return _tool_error(
            "send_reply",
            "not_found",
            f"Ticket '{ticket_id}' not found.",
            correlation_id,
        )

    customer = datastore.customers.get(ticket["customer_email"].lower())
    if customer is None:
        return _tool_error(
            "send_reply",
            "not_found",
            f"Customer for ticket '{ticket_id}' not found.",
            correlation_id,
        )

    first_name = customer["name"].split()[0]
    if first_name.lower() not in clean_message.lower():
        return _tool_error(
            "send_reply",
            "malformed",
            f"Message must address customer by first name ('{first_name}').",
            correlation_id,
        )

    logger.info("[%s] reply sent for %s: %s", correlation_id, ticket_id, clean_message)
    return ReplyResult(ticket_id=ticket_id, message=clean_message, sent_at=datetime.now(timezone.utc))


async def escalate(
    ticket_id: str,
    summary: str,
    priority: PriorityType | str,
    correlation_id: str = "",
) -> EscalateResult | ToolError:
    """
    Escalates a ticket to a human support queue.
    """
    valid_priorities = {"low", "medium", "high", "urgent"}
    normalized_priority = str(priority).lower().strip()
    if normalized_priority not in valid_priorities:
        logger.warning(
            "Invalid escalation priority '%s' for ticket %s. Defaulting to medium.",
            priority,
            ticket_id,
        )
        normalized_priority = "medium"

    if len(summary.strip()) < 50:
        return _tool_error(
            "escalate",
            "malformed",
            "Escalation summary must be at least 50 characters.",
            correlation_id,
        )

    has_order_id = re.search(r"ORD-\d+", summary) is not None
    has_customer_mention = (
        re.search(r"customer|@|C\d{3}", summary, flags=re.IGNORECASE) is not None
    )
    has_action_attempt = (
        re.search(r"verified|checked|attempted|called|refund|tool", summary, flags=re.IGNORECASE)
        is not None
    )

    if not (has_order_id or has_customer_mention or has_action_attempt):
        return _tool_error(
            "escalate",
            "malformed",
            "Summary must include order_id, customer mention, or action attempted.",
            correlation_id,
        )

    team_map = {
        "urgent": ("critical-response", 1),
        "high": ("tier2-priority", 4),
        "medium": ("tier2-general", 12),
        "low": ("tier1-followup", 24),
    }
    team, eta = team_map[normalized_priority]

    return EscalateResult(
        escalation_id=str(uuid.uuid4()),
        ticket_id=ticket_id,
        summary=summary.strip(),
        priority=normalized_priority,  # validated by model
        assigned_team=team,
        estimated_response_hours=eta,
        created_at=datetime.now(timezone.utc),
    )


def reset_runtime_state() -> None:
    """
    Test helper to reset in-memory policy guards and circuit state.
    """
    _eligibility_checked_orders.clear()
    _refunds_issued.clear()
    _failure_injector_holder.instance = None

    circuit_name = "check_refund_eligibility_circuit"
    breaker = CircuitBreakerMonitor.get(circuit_name)
    if breaker is not None:
        breaker.reset()


def get_runtime_sets() -> tuple[set[str], set[str]]:
    """
    Exposes in-memory guard sets for tests.
    """
    return _eligibility_checked_orders, _refunds_issued
