from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from asyncio_throttle import Throttler
from dotenv import load_dotenv

load_dotenv(override=True)

from src.agent import AgentResult, configure_gemini, run_agent
from src.logger import AuditLogger
from src.schemas import TicketRecord
from src.tools import get_datastore


log = structlog.get_logger("shopwave.main")


def validate_environment() -> None:
    """
    Validates all required env vars and data files before processing any ticket.
    """
    required_env = ["GOOGLE_API_KEY"]
    required_files = [
        "data/customers.json",
        "data/tickets.json",
        "data/products.json",
        "data/orders.json",
        "data/knowledge-base.md",
    ]

    errors: list[str] = []
    for variable in required_env:
        if not os.getenv(variable):
            errors.append(f"Missing env var: {variable}")

    for relative_path in required_files:
        if not Path(relative_path).exists():
            errors.append(f"Missing data file: {relative_path}")

    if errors:
        raise EnvironmentError("Startup validation failed:\n" + "\n".join(errors))


def load_tickets(path: str) -> list[dict[str, Any]]:
    ticket_path = Path(path)
    if not ticket_path.exists():
        raise FileNotFoundError(f"Ticket file not found: {path}")

    payload = json.loads(ticket_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tickets.json must contain a JSON array")

    validated: list[dict[str, Any]] = []
    for item in payload:
        record = TicketRecord.model_validate(item)
        validated.append(record.model_dump(mode="json"))

    return validated


def generate_run_summary(results: list[AgentResult]) -> dict[str, Any]:
    total_tickets = len(results)
    resolved = sum(1 for item in results if item.status == "resolved")
    escalated = sum(1 for item in results if item.status == "escalated")
    needs_clarification = sum(1 for item in results if item.status == "needs_clarification")
    failed = sum(1 for item in results if item.status == "failed")
    dead_lettered = sum(1 for item in results if item.status == "dead_lettered")

    avg_processing_time_ms = int(
        sum(item.processing_time_ms for item in results) / total_tickets
    ) if total_tickets else 0

    avg_confidence_score = round(
        sum(item.confidence_score for item in results) / total_tickets, 2
    ) if total_tickets else 0.0

    avg_tool_calls = round(
        sum(item.tool_call_count for item in results) / total_tickets, 2
    ) if total_tickets else 0.0

    tickets_with_retries = sum(1 for item in results if item.retry_count > 0)
    total_retries = sum(item.retry_count for item in results)

    circuit_breaker_trips = 0
    social_engineering_detected = 0
    policy_violations_blocked = 0

    for result in results:
        if any(
            (
                isinstance(tool.get("error"), dict)
                and tool["error"].get("error_type") == "circuit_open"
            )
            for tool in result.tools_called
        ):
            circuit_breaker_trips += 1

        trace_blob = "\n".join(result.reasoning_trace).lower()
        if "social_engineering_detected" in trace_blob:
            social_engineering_detected += 1

        if "policy violation" in (result.error or "").lower() or any(
            tool.get("error") == "PolicyViolationError" for tool in result.tools_called
        ):
            policy_violations_blocked += 1

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "total_tickets": total_tickets,
        "resolved": resolved,
        "escalated": escalated,
        "needs_clarification": needs_clarification,
        "failed": failed,
        "dead_lettered": dead_lettered,
        "avg_processing_time_ms": avg_processing_time_ms,
        "avg_confidence_score": avg_confidence_score,
        "avg_tool_calls_per_ticket": avg_tool_calls,
        "tickets_with_retries": tickets_with_retries,
        "total_retries": total_retries,
        "circuit_breaker_trips": circuit_breaker_trips,
        "social_engineering_detected": social_engineering_detected,
        "policy_violations_blocked": policy_violations_blocked,
    }


def write_run_summary(summary: dict[str, Any], path: str = "run_summary.json") -> None:
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def print_run_summary(summary: dict[str, Any]) -> None:
    printable_keys = [
        "total_tickets",
        "resolved",
        "escalated",
        "needs_clarification",
        "failed",
        "dead_lettered",
        "avg_processing_time_ms",
        "avg_confidence_score",
        "avg_tool_calls_per_ticket",
        "tickets_with_retries",
        "total_retries",
        "circuit_breaker_trips",
    ]
    log.info("run_summary_ready", **{key: summary.get(key) for key in printable_keys})


async def main() -> None:
    validate_environment()
    get_datastore()

    client = configure_gemini()
    audit_logger = AuditLogger()

    tickets = load_tickets("data/tickets.json")
    log.info("tickets_loaded", count=len(tickets))

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "2"))
    throttler = Throttler(rate_limit=max_concurrency)
    ticket_stagger_seconds = float(os.getenv("TICKET_STAGGER_SECONDS", "1.0"))

    async def process_with_throttle(ticket: dict[str, Any], index: int) -> AgentResult:
        async with throttler:
            if ticket_stagger_seconds > 0:
                await asyncio.sleep(index * ticket_stagger_seconds)
            ticket_id = ticket.get("ticket_id", "unknown")
            log.info("ticket_processing_start", ticket_id=ticket_id)

            result = await run_agent(ticket, client)
            await audit_logger.append_result(result)

            if result.status == "dead_lettered":
                await audit_logger.append_dead_letter(
                    ticket=ticket,
                    error_chain=result.reasoning_trace + ([result.error] if result.error else []),
                    correlation_id=result.correlation_id,
                )

            log.info(
                "ticket_processing_complete",
                ticket_id=ticket_id,
                status=result.status,
                time_ms=result.processing_time_ms,
            )
            return result

    raw_results = await asyncio.gather(
        *[process_with_throttle(ticket, idx) for idx, ticket in enumerate(tickets)],
        return_exceptions=True,
    )

    results: list[AgentResult] = []
    for index, raw_result in enumerate(raw_results):
        if isinstance(raw_result, Exception):
            ticket = tickets[index]
            ticket_id = ticket.get("ticket_id", "unknown")
            correlation_id = "gather-exception"
            error_message = f"Unhandled exception in gather: {type(raw_result).__name__}: {raw_result}"
            log.error("ticket_gather_exception", ticket_id=ticket_id, error=error_message)

            failed_result = AgentResult(
                ticket_id=ticket_id,
                correlation_id=correlation_id,
                customer_email=ticket.get("customer_email", ""),
                status="failed",
                final_action="unknown",
                error=error_message,
            )
            results.append(failed_result)

            await audit_logger.append_result(failed_result)
            await audit_logger.append_dead_letter(
                ticket=ticket,
                error_chain=[error_message],
                correlation_id=correlation_id,
            )
        else:
            results.append(raw_result)

    summary = generate_run_summary(results)
    write_run_summary(summary)
    print_run_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
