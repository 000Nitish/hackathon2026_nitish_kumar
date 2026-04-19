from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import google.genai as genai
from google.genai import types
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .prompts import SYSTEM_PROMPT
from .schemas import ToolError
from .tools import (
    PolicyViolationError,
    check_refund_eligibility,
    escalate,
    get_customer,
    get_order,
    get_product,
    issue_refund,
    search_knowledge_base,
    send_reply,
)


logger = logging.getLogger("shopwave.agent")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


MODEL_ID_DEFAULT = "gemini-3-flash-preview"
CONFIDENCE_ESCALATION_THRESHOLD = float(os.getenv("CONFIDENCE_ESCALATION_THRESHOLD", "0.60"))
MAX_GEMINI_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
GEMINI_RETRY_WAIT_MIN = int(os.getenv("GEMINI_RETRY_WAIT_MIN", "15"))
GEMINI_RETRY_WAIT_MAX = int(os.getenv("GEMINI_RETRY_WAIT_MAX", "120"))
GEMINI_REQUESTS_PER_MINUTE = max(1, int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "5")))
GEMINI_RATE_WINDOW_SECONDS = max(
    1.0, float(os.getenv("GEMINI_RATE_WINDOW_SECONDS", "60"))
)
MINIMUM_TOOL_CHAIN_MESSAGE = (
    "SYSTEM: Minimum tool chain not satisfied. You must call at least one more verification tool before concluding."
)
TERMINAL_TOOLS: set[str] = {"send_reply", "escalate"}
VERIFICATION_TOOLS: set[str] = {
    "get_customer",
    "get_order",
    "get_product",
    "search_knowledge_base",
    "check_refund_eligibility",
    "issue_refund",
}


GENERATION_TEMPERATURE = 0.1
GENERATION_MAX_OUTPUT_TOKENS = 4096
GENERATION_TOP_P = 0.95
GENERATION_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "8000"))


TOOL_DECLARATIONS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_order",
                description=(
                    "Retrieve full order details by order ID. Call this early for any ticket that references "
                    "an order. Returns status, dates, amount, product ID, and notes. If timeout occurs, "
                    "the platform retries automatically before escalation is considered."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "order_id": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Order ID in format ORD-XXXX (example: ORD-1001). Extract from ticket text. "
                                "If missing, do not guess and gather more details first."
                            ),
                        )
                    },
                    required=["order_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_customer",
                description=(
                    "Retrieve customer profile and verified tier by registered email. This must be the first "
                    "tool call for every ticket. Tier in this response is authoritative for policy decisions."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "email": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Customer email address from ticket metadata (example: name@example.com). "
                                "Pass exact value; do not infer an alternate address."
                            ),
                        )
                    },
                    required=["email"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_product",
                description=(
                    "Retrieve product metadata using product_id from an order. Returns category, return window, "
                    "warranty months, and returnability constraints used for policy checks."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "product_id": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Product ID in format PXXX (example: P001). Usually obtained from get_order response."
                            ),
                        )
                    },
                    required=["product_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_knowledge_base",
                description=(
                    "Search policy and operations guidance. Use targeted queries such as 'refund eligibility', "
                    "'warranty claim process', or 'cancellation shipped order' and cite the relevant section "
                    "in Thought before next action."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Natural-language policy query with key details (issue type, tier nuance, order state)."
                            ),
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="check_refund_eligibility",
                description=(
                    "Evaluate whether an order qualifies for refund under delivery status, refund history, return window, "
                    "and amount threshold rules. Must be called before any issue_refund attempt."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "order_id": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Order ID in format ORD-XXXX. Provide the exact order under review for refund decision."
                            ),
                        )
                    },
                    required=["order_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="issue_refund",
                description=(
                    "Issue an irreversible refund after eligibility is confirmed. Use only when the order is eligible "
                    "and amount does not require escalation."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "order_id": types.Schema(
                            type=types.Type.STRING,
                            description="Order ID that was already validated via check_refund_eligibility.",
                        ),
                        "amount": types.Schema(
                            type=types.Type.NUMBER,
                            description=(
                                "Refund amount as positive decimal. Must not exceed the original order amount with "
                                "floating-point tolerance."
                            ),
                        ),
                    },
                    required=["order_id", "amount"],
                ),
            ),
            types.FunctionDeclaration(
                name="send_reply",
                description=(
                    "Send the final customer-facing response. Message must be over 20 characters and include the "
                    "customer first name. Use only after sufficient verification chain."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "ticket_id": types.Schema(
                            type=types.Type.STRING,
                            description="Ticket ID in format TKT-XXX (example: TKT-001).",
                        ),
                        "message": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Final response text: acknowledge issue, state decision, explain reason, and provide "
                                "next step/timeline while addressing customer by first name."
                            ),
                        ),
                    },
                    required=["ticket_id", "message"],
                ),
            ),
            types.FunctionDeclaration(
                name="escalate",
                description=(
                    "Escalate to a human team with concise but complete context. Required for warranty claims, "
                    "high-risk conditions, low confidence, fraud concerns, or circuit-open tool paths."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "ticket_id": types.Schema(
                            type=types.Type.STRING,
                            description="Ticket ID in format TKT-XXX.",
                        ),
                        "summary": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Detailed summary (50+ chars) including issue, verified facts, attempted actions, "
                                "recommended path, and reason autonomous resolution cannot proceed."
                            ),
                        ),
                        "priority": types.Schema(
                            type=types.Type.STRING,
                            description="One of: low, medium, high, urgent.",
                            enum=["low", "medium", "high", "urgent"],
                        ),
                    },
                    required=["ticket_id", "summary", "priority"],
                ),
            ),
        ]
    )
]


ToolCallable = Callable[..., Awaitable[Any]]
TOOL_REGISTRY: dict[str, ToolCallable] = {
    "get_order": get_order,
    "get_customer": get_customer,
    "get_product": get_product,
    "search_knowledge_base": search_knowledge_base,
    "check_refund_eligibility": check_refund_eligibility,
    "issue_refund": issue_refund,
    "send_reply": send_reply,
    "escalate": escalate,
}


_dead_letter_lock = asyncio.Lock()


class GeminiRateLimiter:
    """
    Sliding-window request limiter for Gemini API calls.
    Prevents quota bursts across concurrent ticket workers.
    """

    def __init__(self, requests_per_window: int, window_seconds: float) -> None:
        self.requests_per_window = max(1, requests_per_window)
        self.window_seconds = max(1.0, window_seconds)
        self._lock = asyncio.Lock()
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            wait_for = 0.0
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and (now - self._timestamps[0]) >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.requests_per_window:
                    self._timestamps.append(now)
                    return

                oldest = self._timestamps[0]
                wait_for = max(0.01, self.window_seconds - (now - oldest) + 0.01)

            await asyncio.sleep(wait_for)


_gemini_rate_limiter = GeminiRateLimiter(
    requests_per_window=GEMINI_REQUESTS_PER_MINUTE,
    window_seconds=GEMINI_RATE_WINDOW_SECONDS,
)


@dataclass
class AgentResult:
    ticket_id: str
    correlation_id: str
    customer_email: str
    status: Literal["resolved", "escalated", "failed", "needs_clarification", "dead_lettered"]
    final_action: Literal[
        "refund_issued",
        "escalated",
        "reply_sent",
        "cancelled",
        "exchange_initiated",
        "needs_clarification",
        "dead_lettered",
        "unknown",
    ]
    reasoning_trace: list[str] = field(default_factory=list)
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    tool_call_count: int = 0
    retry_count: int = 0
    error: str | None = None
    customer_name: str | None = None
    order_id: str | None = None


def configure_gemini() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


def _build_generation_config(model_id: str) -> types.GenerateContentConfig:
    supports_thinking = "2.5" in model_id or "thinking" in model_id
    thinking_config = (
        types.ThinkingConfig(thinking_budget=GENERATION_THINKING_BUDGET)
        if supports_thinking
        else None
    )
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=TOOL_DECLARATIONS,
        temperature=GENERATION_TEMPERATURE,
        max_output_tokens=GENERATION_MAX_OUTPUT_TOKENS,
        top_p=GENERATION_TOP_P,
        thinking_config=thinking_config,
    )


def format_ticket_for_agent(ticket: dict[str, Any]) -> str:
    return (
        "Please resolve the following customer support ticket using the provided tools.\n\n"
        f"ticket_id: {ticket.get('ticket_id', '')}\n"
        f"customer_email: {ticket.get('customer_email', '')}\n"
        f"subject: {ticket.get('subject', '')}\n"
        f"body: {ticket.get('body', '')}\n"
        f"source: {ticket.get('source', '')}\n"
        f"created_at: {ticket.get('created_at', '')}\n"
        "\nRequirements reminder:\n"
        "- Emit Thought with [STATE] and [POLICY] before each tool call\n"
        "- Verify customer first\n"
        "- Minimum 3 distinct verification tools before send_reply/escalate\n"
        "- Assign confidence_score before final action\n"
    )


def extract_final_response(response: types.GenerateContentResponse) -> str:
    candidate = _primary_candidate(response)
    if candidate is None:
        return ""

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    text_chunks: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if part_text:
            text_chunks.append(part_text.strip())
    return "\n".join(chunk for chunk in text_chunks if chunk)


def infer_status_from_response(final_text: str, result: AgentResult) -> Literal[
    "resolved", "escalated", "failed", "needs_clarification", "dead_lettered"
]:
    if result.status in {"resolved", "escalated", "needs_clarification", "dead_lettered"}:
        return result.status

    lowered = final_text.lower()
    if result.final_action == "escalated" or "escalat" in lowered:
        return "escalated"
    if result.final_action == "dead_lettered":
        return "dead_lettered"
    if result.final_action in {"needs_clarification"} or "clarif" in lowered:
        return "needs_clarification"
    if result.final_action in {"refund_issued", "reply_sent", "cancelled", "exchange_initiated"}:
        return "resolved"
    if "resolved" in lowered or "refund" in lowered or "approved" in lowered:
        return "resolved"
    return "failed"


def _primary_candidate(response: types.GenerateContentResponse) -> Any | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return candidates[0]


def _extract_function_calls(candidate: Any | None) -> list[Any]:
    if candidate is None:
        return []
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    function_calls: list[Any] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            function_calls.append(function_call)
    return function_calls


def _extract_thought_lines(candidate: Any | None) -> list[str]:
    if candidate is None:
        return []
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []

    lines: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if not part_text:
            continue
        for raw_line in part_text.splitlines():
            line = raw_line.strip()
            if line.startswith("Thought:"):
                lines.append(line)
    return lines


def _function_args_to_dict(function_call: Any) -> dict[str, Any]:
    args = getattr(function_call, "args", None)
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)

    try:
        return dict(args)
    except Exception:
        try:
            return json.loads(str(args))
        except Exception:
            logger.warning(
                "Unable to parse function arguments for call '%s'",
                getattr(function_call, "name", ""),
            )
            return {}


def _extract_confidence_score(text: str) -> float:
    matches = re.findall(r"confidence_score\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not matches:
        return 0.0

    try:
        score = float(matches[-1])
    except ValueError:
        return 0.0

    return max(0.0, min(1.0, score))


def _next_state(current_state: str, tool_name: str, tool_payload: dict[str, Any]) -> str:
    if tool_name == "get_customer":
        if "error_type" in tool_payload and tool_payload["error_type"] == "not_found":
            return "NEEDS_CLARIFICATION"
        return "CUSTOMER_VERIFIED"

    if tool_name == "get_order":
        if "error_type" in tool_payload and tool_payload["error_type"] == "not_found":
            return "NEEDS_CLARIFICATION"
        return "ORDER_VERIFIED"

    if tool_name == "search_knowledge_base":
        return "POLICY_CHECKED"

    if tool_name == "check_refund_eligibility":
        return "ELIGIBILITY_VERIFIED"

    if tool_name == "issue_refund":
        return "ACTION_TAKEN"

    if tool_name == "escalate":
        return "ESCALATING"

    if tool_name == "send_reply":
        return "ACTION_TAKEN"

    return current_state


def _append_state_trace(result: AgentResult, state: str, policy: str, reason: str) -> None:
    result.reasoning_trace.append(f"Thought: [STATE: {state}] [POLICY: {policy}] {reason}")


def _classification_from_reply(message: str) -> Literal[
    "reply_sent", "cancelled", "exchange_initiated", "needs_clarification"
]:
    lowered = message.lower()
    if "clarif" in lowered or "please provide" in lowered or "order number" in lowered:
        return "needs_clarification"
    if "cancel" in lowered and ("confirm" in lowered or "cancelled" in lowered):
        return "cancelled"
    if "exchange" in lowered or "replacement" in lowered:
        return "exchange_initiated"
    return "reply_sent"


def _extract_order_id_from_ticket(ticket: dict[str, Any]) -> str | None:
    blob = f"{ticket.get('subject', '')} {ticket.get('body', '')}"
    match = re.search(r"\bORD-\d+\b", blob, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(0).upper()


def _is_model_unavailable_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "clienterror" in name and (
        "resource_exhausted" in text
        or "quota exceeded" in text
        or "429" in text
        or "rate limit" in text
    )


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """
    True for transient/rate limit Gemini failures.
    False for permanent auth/request failures.
    """
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if any(token in text for token in ("429", "resource_exhausted", "quota", "rate_limit", "503", "unavailable")):
        return True
    if any(
        token in text
        for token in (
            "400",
            "401",
            "403",
            "404",
            "not_found",
            "not found",
            "invalid_argument",
            "permission_denied",
        )
    ):
        return False
    return "error" in name or "exception" in name


def _extract_retry_after_seconds(exc: Exception) -> float:
    text = str(exc)
    lowered = text.lower()

    match = re.search(r"retry in\s+(\d+(?:\.\d+)?)s", lowered)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            return 0.0

    match = re.search(r"'retrydelay':\s*'(\d+)s'", lowered)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            return 0.0

    return 0.0


def _log_gemini_retry_attempt(state: RetryCallState) -> None:
    error = state.outcome.exception() if state.outcome else None
    wait_seconds = getattr(state.next_action, "sleep", 0)
    logger.warning(
        "gemini_api_retry attempt=%s wait=%ss error=%s",
        state.attempt_number,
        wait_seconds,
        str(error),
    )


def _gemini_retry_exhausted(_state: RetryCallState) -> None:
    return None


@retry(
    stop=stop_after_attempt(MAX_GEMINI_RETRIES),
    wait=wait_exponential(multiplier=2, min=GEMINI_RETRY_WAIT_MIN, max=GEMINI_RETRY_WAIT_MAX),
    retry=retry_if_exception(_is_retryable_gemini_error),
    before_sleep=_log_gemini_retry_attempt,
    retry_error_callback=_gemini_retry_exhausted,
    reraise=False,
)
async def _call_gemini_with_retry(
    client: genai.Client,
    model_id: str,
    conversation_history: list[types.Content],
    generation_config: types.GenerateContentConfig,
) -> types.GenerateContentResponse | None:
    await _gemini_rate_limiter.acquire()
    try:
        return await asyncio.to_thread(
            client.models.generate_content,
            model=model_id,
            contents=conversation_history,
            config=generation_config,
        )
    except Exception as exc:
        if not _is_retryable_gemini_error(exc):
            raise
        retry_after_seconds = _extract_retry_after_seconds(exc)
        if retry_after_seconds > 0:
            await asyncio.sleep(retry_after_seconds)
        raise


async def _handle_model_unavailable(
    ticket: dict[str, Any],
    result: AgentResult,
    correlation_id: str,
    current_state: str,
    error_message: str,
) -> None:
    """
    Fallback path when Gemini API is unavailable (quota/rate limit).
    Performs a compliant verification chain and escalates with context.
    """
    _append_state_trace(
        result,
        state=current_state,
        policy="Escalation Guidelines",
        reason=(
            "Gemini API unavailable due to quota/rate limits. "
            "Switching to deterministic fallback verification before escalation."
        ),
    )

    verification_tool_set: set[str] = set()

    _append_state_trace(
        result,
        state="INGESTING",
        policy="Customer Tiers & Privileges",
        reason="Calling get_customer first to verify identity and tier before any escalation.",
    )
    customer_payload = await execute_tool_safely(
        tool_name="get_customer",
        args={"email": ticket.get("customer_email", "")},
        result=result,
        correlation_id=correlation_id,
    )
    verification_tool_set.add("get_customer")
    current_state = _next_state(current_state, "get_customer", customer_payload)

    order_id = _extract_order_id_from_ticket(ticket)
    if order_id:
        _append_state_trace(
            result,
            state=current_state,
            policy="Return Policy - Standard Return Window",
            reason=f"Order reference {order_id} found in ticket. Calling get_order for objective order facts.",
        )
        order_payload = await execute_tool_safely(
            tool_name="get_order",
            args={"order_id": order_id},
            result=result,
            correlation_id=correlation_id,
        )
        verification_tool_set.add("get_order")
        current_state = _next_state(current_state, "get_order", order_payload)

        product_id = order_payload.get("product_id") if isinstance(order_payload, dict) else None
        if isinstance(product_id, str) and product_id:
            _append_state_trace(
                result,
                state=current_state,
                policy="Return Policy - Category-Specific Return Windows",
                reason=(
                    f"Order maps to product {product_id}. Calling get_product to preserve "
                    "category- and warranty-specific policy context."
                ),
            )
            await execute_tool_safely(
                tool_name="get_product",
                args={"product_id": product_id},
                result=result,
                correlation_id=correlation_id,
            )
            verification_tool_set.add("get_product")
            current_state = _next_state(current_state, "get_product", {"product_id": product_id})
        else:
            _append_state_trace(
                result,
                state=current_state,
                policy="Escalation Guidelines",
                reason=(
                    "Product details are unavailable. Calling search_knowledge_base to include "
                    "relevant policy context in escalation."
                ),
            )
            await execute_tool_safely(
                tool_name="search_knowledge_base",
                args={"query": "escalation when product context missing and model unavailable"},
                result=result,
                correlation_id=correlation_id,
            )
            verification_tool_set.add("search_knowledge_base")
            current_state = _next_state(current_state, "search_knowledge_base", {})
    else:
        _append_state_trace(
            result,
            state=current_state,
            policy="Ambiguous Tickets - Clarification Protocol",
            reason=(
                "No explicit order ID detected. Calling search_knowledge_base to gather "
                "clarification and escalation guidance."
            ),
        )
        await execute_tool_safely(
            tool_name="search_knowledge_base",
            args={"query": "clarification needed missing order details"},
            result=result,
            correlation_id=correlation_id,
        )
        verification_tool_set.add("search_knowledge_base")
        current_state = _next_state(current_state, "search_knowledge_base", {})

    if len(verification_tool_set) < 3:
        _append_state_trace(
            result,
            state=current_state,
            policy="Escalation Guidelines",
            reason=(
                "Minimum verification chain not yet met in fallback path. "
                "Calling search_knowledge_base for additional policy verification."
            ),
        )
        await execute_tool_safely(
            tool_name="search_knowledge_base",
            args={"query": "policy checks before escalation due to system outage"},
            result=result,
            correlation_id=correlation_id,
        )
        verification_tool_set.add("search_knowledge_base")
        current_state = _next_state(current_state, "search_knowledge_base", {})

    successful_verifications = sorted(
        {
            entry.get("tool", "")
            for entry in result.tools_called
            if entry.get("success") and entry.get("tool") in VERIFICATION_TOOLS
        }
    )

    _append_state_trace(
        result,
        state="ESCALATING",
        policy="Escalation Guidelines",
        reason=(
            "Escalating because autonomous model is unavailable. "
            "Fallback verification chain is complete."
        ),
    )

    escalation_summary = (
        f"Autonomous model unavailable due to API quota/rate limits ({error_message[:180]}). "
        f"Ticket {result.ticket_id} for customer {result.customer_email}. "
        f"Verified via tools: {', '.join(successful_verifications) or 'none'}. "
        "Recommended path: human agent continuation using verified context."
    )
    escalation_payload = await execute_tool_safely(
        tool_name="escalate",
        args={
            "ticket_id": result.ticket_id,
            "summary": escalation_summary,
            "priority": "high",
        },
        result=result,
        correlation_id=correlation_id,
    )

    if escalation_payload.get("error"):
        result.status = "failed"
        result.final_action = "unknown"
        result.error = f"Fallback escalation failed: {escalation_payload.get('error')}"
        result.confidence_score = 0.0
    else:
        result.status = "escalated"
        result.final_action = "escalated"
        result.confidence_score = 0.0
        result.error = "LLM_UNAVAILABLE: API retries exhausted. Resolved via deterministic fallback."


async def move_to_dead_letter(ticket: dict[str, Any], error_chain: list[str], correlation_id: str) -> None:
    """
    Writes unresolved tickets with context into dead_letter.json.
    """
    root_path = Path(__file__).resolve().parent.parent
    target_path = root_path / "dead_letter.json"

    payload = {
        "ticket_id": ticket.get("ticket_id", "unknown"),
        "correlation_id": correlation_id,
        "customer_email": ticket.get("customer_email", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error_chain": error_chain,
        "ticket": ticket,
    }

    async with _dead_letter_lock:
        if not target_path.exists():
            await asyncio.to_thread(target_path.write_text, "[]", "utf-8")

        try:
            existing_raw = await asyncio.to_thread(target_path.read_text, "utf-8")
            existing_data = json.loads(existing_raw)
            if not isinstance(existing_data, list):
                existing_data = []
        except Exception as exc:
            logger.exception("Failed to parse dead_letter.json. Reinitializing.", exc_info=exc)
            existing_data = []

        existing_data.append(payload)
        await asyncio.to_thread(target_path.write_text, json.dumps(existing_data, indent=2), "utf-8")


async def run_agent(ticket: dict[str, Any], client: genai.Client) -> AgentResult:
    """
    Executes the ReAct loop for one ticket.
    """
    correlation_id = str(uuid.uuid4())[:8]
    result = AgentResult(
        ticket_id=ticket.get("ticket_id", "unknown"),
        correlation_id=correlation_id,
        customer_email=ticket.get("customer_email", ""),
        status="failed",
        final_action="unknown",
    )

    start_time = time.monotonic()
    model_id = os.getenv("GEMINI_MODEL", MODEL_ID_DEFAULT)
    max_iterations = int(os.getenv("MAX_REACT_ITERATIONS", "10"))
    initial_message = format_ticket_for_agent(ticket)

    conversation_history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=initial_message)])
    ]
    current_state = "INGESTING"
    verification_tool_set: set[str] = set()

    _append_state_trace(
        result,
        state=current_state,
        policy="System State Machine",
        reason="Ticket ingested; starting verification workflow.",
    )

    try:
        for iteration in range(max_iterations):
            response = await _call_gemini_with_retry(
                client=client,
                model_id=model_id,
                conversation_history=conversation_history,
                generation_config=_build_generation_config(model_id=model_id),
            )

            if response is None:
                logger.warning(
                    "gemini_retries_exhausted",
                    extra={
                        "ticket_id": result.ticket_id,
                        "correlation_id": correlation_id,
                        "iteration": iteration,
                    },
                )
                await _handle_model_unavailable(
                    ticket=ticket,
                    result=result,
                    correlation_id=correlation_id,
                    current_state=current_state,
                    error_message="All Gemini API retries exhausted (rate limit or quota)",
                )
                break

            candidate = _primary_candidate(response)
            if candidate is None:
                result.error = "Gemini response missing candidates"
                logger.error(
                    "model_response_missing_candidates",
                    extra={
                        "ticket_id": result.ticket_id,
                        "correlation_id": correlation_id,
                        "iteration": iteration,
                    },
                )
                break

            conversation_history.append(candidate.content)

            thought_lines = _extract_thought_lines(candidate)
            result.reasoning_trace.extend(thought_lines)

            function_calls = _extract_function_calls(candidate)
            if not function_calls:
                final_text = extract_final_response(response)
                if final_text:
                    result.reasoning_trace.append(final_text.strip())
                score = _extract_confidence_score("\n".join(result.reasoning_trace))
                if score > 0:
                    result.confidence_score = score

                if (
                    result.confidence_score == 0.0
                    or result.confidence_score < CONFIDENCE_ESCALATION_THRESHOLD
                ) and result.status not in {
                    "escalated",
                    "dead_lettered",
                }:
                    escalate_summary = (
                        f"confidence_score={result.confidence_score:.2f} below threshold "
                        f"{CONFIDENCE_ESCALATION_THRESHOLD:.2f}. Verified tools: "
                        f"{', '.join(sorted(verification_tool_set)) or 'none'}."
                    )
                    escalation = await escalate(
                        ticket_id=result.ticket_id,
                        summary=escalate_summary + " Escalating due to confidence policy.",
                        priority="medium",
                        correlation_id=correlation_id,
                    )
                    if isinstance(escalation, ToolError):
                        result.error = escalation.message
                        result.status = "failed"
                    else:
                        result.status = "escalated"
                        result.final_action = "escalated"

                if result.status == "failed":
                    result.status = infer_status_from_response(final_text, result)
                break

            tool_parts: list[types.Part] = []
            turn_all_circuit_open = True

            for function_call in function_calls:
                tool_name = str(getattr(function_call, "name", "")).strip()
                args = _function_args_to_dict(function_call)

                if tool_name in TERMINAL_TOOLS and len(verification_tool_set) < 3:
                    blocked_observation = {
                        "error": MINIMUM_TOOL_CHAIN_MESSAGE,
                        "error_type": "policy_violation",
                    }
                    tool_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": blocked_observation},
                            )
                        )
                    )
                    result.tools_called.append(
                        {
                            "tool": tool_name,
                            "args": args,
                            "success": False,
                            "retries": 0,
                            "error": blocked_observation,
                            "blocked_by_system": True,
                        }
                    )
                    _append_state_trace(
                        result,
                        state=current_state,
                        policy="System Guard - Minimum Tool Chain",
                        reason=(
                            f"Blocked {tool_name} because only {len(verification_tool_set)} "
                            "distinct verification tools were called."
                        ),
                    )
                    turn_all_circuit_open = False
                    continue

                tool_result = await execute_tool_safely(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    correlation_id=correlation_id,
                )

                if tool_name in VERIFICATION_TOOLS:
                    verification_tool_set.add(tool_name)

                if not (tool_result.get("error_type") == "circuit_open"):
                    turn_all_circuit_open = False

                next_state = _next_state(current_state, tool_name, tool_result)
                if next_state != current_state:
                    _append_state_trace(
                        result,
                        state=next_state,
                        policy="System State Machine",
                        reason=f"Transitioned from {current_state} to {next_state} after {tool_name}.",
                    )
                    current_state = next_state

                tool_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": tool_result},
                        )
                    )
                )

            conversation_history.append(types.Content(role="user", parts=tool_parts))

            if turn_all_circuit_open and function_calls:
                _append_state_trace(
                    result,
                    state="ESCALATING",
                    policy="Escalation Guidelines",
                    reason="Critical tools returned circuit_open; escalation required.",
                )
                escalation = await escalate(
                    ticket_id=result.ticket_id,
                    summary=(
                        "Critical tool path unavailable due to open circuit state. "
                        f"Ticket {result.ticket_id} has repeated circuit-open failures and requires human handling."
                    ),
                    priority="high",
                    correlation_id=correlation_id,
                )
                if isinstance(escalation, ToolError):
                    result.error = f"Circuit-open escalation failed: {escalation.message}"
                    result.status = "failed"
                else:
                    result.status = "escalated"
                    result.final_action = "escalated"
                break

            if result.status in {"resolved", "escalated", "needs_clarification", "dead_lettered"}:
                break

        else:
            await escalate(
                ticket_id=result.ticket_id,
                summary=(
                    f"Max iterations ({max_iterations}) reached without resolution. "
                    f"Last trace items: {'; '.join(result.reasoning_trace[-3:])}"
                ),
                priority="medium",
                correlation_id=correlation_id,
            )
            result.status = "escalated"
            result.final_action = "escalated"

    except PolicyViolationError as exc:
        result.status = "dead_lettered"
        result.final_action = "dead_lettered"
        result.error = str(exc)
        logger.error(
            "policy_violation_dead_letter",
            extra={
                "ticket_id": result.ticket_id,
                "correlation_id": correlation_id,
                "tool_name": exc.tool_name,
                "order_id": exc.order_id,
                "error": str(exc),
            },
        )
        await move_to_dead_letter(ticket, result.reasoning_trace + [str(exc)], correlation_id)

    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {str(exc)}"
        logger.exception(
            "agent_unexpected_error",
            extra={
                "ticket_id": result.ticket_id,
                "correlation_id": correlation_id,
                "error": str(exc),
            },
            exc_info=exc,
        )

    finally:
        if result.confidence_score <= 0:
            result.confidence_score = _extract_confidence_score("\n".join(result.reasoning_trace))
        result.processing_time_ms = int((time.monotonic() - start_time) * 1000)

    return result


async def execute_tool_safely(
    tool_name: str,
    args: dict[str, Any],
    result: AgentResult,
    correlation_id: str,
) -> dict[str, Any]:
    """
    Executes a single tool call with error handling.
    """
    result.tool_call_count += 1

    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        return {
            "error": f"Unknown tool: {tool_name}. Available tools: {sorted(TOOL_REGISTRY.keys())}",
            "error_type": "malformed",
        }

    call_args = dict(args)
    call_args["correlation_id"] = correlation_id
    tool_entry: dict[str, Any] = {
        "tool": tool_name,
        "args": call_args,
        "success": False,
        "retries": 0,
    }

    try:
        tool_result = await tool_fn(**call_args)

        if isinstance(tool_result, ToolError):
            payload = tool_result.model_dump(mode="json")
            tool_entry["error"] = payload
            tool_entry["retries"] = tool_result.retries_attempted
            result.retry_count += tool_result.retries_attempted
            result.tools_called.append(tool_entry)
            return {
                "error": tool_result.message,
                "error_type": tool_result.error_type,
                "tool_name": tool_name,
            }

        model_payload = (
            tool_result.model_dump(mode="json")
            if hasattr(tool_result, "model_dump")
            else {"result": str(tool_result)}
        )

        tool_entry["success"] = True
        result.tools_called.append(tool_entry)

        if tool_name == "get_customer":
            result.customer_name = str(model_payload.get("name") or "") or result.customer_name

        if tool_name in {"get_order", "check_refund_eligibility", "issue_refund"}:
            order_id = model_payload.get("order_id")
            if isinstance(order_id, str) and order_id:
                result.order_id = order_id

        if tool_name == "issue_refund":
            result.final_action = "refund_issued"
            if result.status not in {"escalated", "dead_lettered"}:
                result.status = "resolved"

        if tool_name == "send_reply":
            message = str(model_payload.get("message", ""))
            classification = _classification_from_reply(message)
            result.final_action = classification
            if classification == "needs_clarification":
                result.status = "needs_clarification"
            elif result.status not in {"escalated", "dead_lettered"}:
                result.status = "resolved"

        if tool_name == "escalate":
            result.final_action = "escalated"
            result.status = "escalated"

        return model_payload

    except PolicyViolationError:
        tool_entry["error"] = "PolicyViolationError"
        result.tools_called.append(tool_entry)
        raise

    except Exception as exc:
        tool_entry["error"] = str(exc)
        result.tools_called.append(tool_entry)
        logger.exception(
            "tool_unexpected_error",
            extra={
                "ticket_id": result.ticket_id,
                "correlation_id": correlation_id,
                "tool_name": tool_name,
                "error": str(exc),
            },
            exc_info=exc,
        )
        return {
            "error": f"Unexpected error in {tool_name}: {str(exc)}",
            "error_type": "malformed",
            "tool_name": tool_name,
        }
