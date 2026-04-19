from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORE_WEIGHTS: dict[str, int] = {
    "correct_action": 40,
    "tool_chain_depth": 20,
    "policy_cited": 20,
    "customer_named": 10,
    "confidence_scored": 10,
}


def _load_json(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def _expected_action_bucket(expected_text: str) -> str:
    """
    Maps expected_action text to resolution bucket.
    Priority order avoids incidental keyword false positives.
    """
    lowered = expected_text.lower()

    if "clarif" in lowered or ("ask" in lowered and "order" in lowered):
        return "needs_clarification"

    warranty_escalate = "warranty claim" in lowered or "escalate as warranty" in lowered
    explicit_escalate = "escalat" in lowered and "refund" not in lowered
    if warranty_escalate or explicit_escalate:
        return "escalated"

    if "social engineering" in lowered or ("flag" in lowered and "premium" in lowered):
        return "escalated"

    if "cancel" in lowered and ("immediately" in lowered or "confirm" in lowered):
        return "cancelled"

    if ("exchange" in lowered or "replace" in lowered) and "refund" not in lowered:
        return "exchange_initiated"

    if (
        "issue refund" in lowered
        or "full refund" in lowered
        or "approve return" in lowered
        or "refund" in lowered
    ):
        return "refund_issued"

    return "reply_sent"


def _actual_action_bucket(audit_entry: dict[str, Any]) -> str:
    final_action = str(audit_entry.get("final_action", "unknown")).lower()
    status = str(audit_entry.get("status", "failed")).lower()

    if status == "needs_clarification":
        return "needs_clarification"
    if status == "escalated" or final_action == "escalated":
        return "escalated"
    if final_action in {"refund_issued", "cancelled", "exchange_initiated", "needs_clarification"}:
        return final_action
    if final_action in {"reply_sent"}:
        return "reply_sent"
    if status == "resolved":
        return "reply_sent"
    return "unknown"


def _find_sent_reply_message(audit_entry: dict[str, Any]) -> str:
    tools_called = audit_entry.get("tools_called", [])
    if not isinstance(tools_called, list):
        return ""

    for call in tools_called:
        if not isinstance(call, dict):
            continue
        if call.get("tool") != "send_reply":
            continue
        args = call.get("args", {})
        if isinstance(args, dict):
            message = args.get("message", "")
            if isinstance(message, str):
                return message
    return ""


def score_ticket(audit_entry: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_bucket = _expected_action_bucket(str(expected.get("expected_action", "")))
    actual_bucket = _actual_action_bucket(audit_entry)

    score_correct_action = SCORE_WEIGHTS["correct_action"] if expected_bucket == actual_bucket else 0

    tool_call_count = int(audit_entry.get("tool_call_count", 0) or 0)
    score_tool_chain_depth = SCORE_WEIGHTS["tool_chain_depth"] if tool_call_count >= 3 else 0

    trace = audit_entry.get("reasoning_trace", [])
    trace_text = "\n".join(trace) if isinstance(trace, list) else str(trace)
    score_policy_cited = (
        SCORE_WEIGHTS["policy_cited"]
        if "Thought:" in trace_text and "[POLICY:" in trace_text
        else 0
    )

    message = _find_sent_reply_message(audit_entry)
    customer_name = str(audit_entry.get("customer_name", "")).strip().split(" ")[0]
    if actual_bucket in {"reply_sent", "cancelled", "exchange_initiated", "needs_clarification"}:
        if customer_name and customer_name.lower() in message.lower():
            score_customer_named = SCORE_WEIGHTS["customer_named"]
        else:
            score_customer_named = 0
    else:
        score_customer_named = SCORE_WEIGHTS["customer_named"]

    confidence_score = float(audit_entry.get("confidence_score", 0.0) or 0.0)
    score_confidence = SCORE_WEIGHTS["confidence_scored"] if confidence_score > 0 else 0

    total = (
        score_correct_action
        + score_tool_chain_depth
        + score_policy_cited
        + score_customer_named
        + score_confidence
    )

    return {
        "ticket_id": expected.get("ticket_id", audit_entry.get("ticket_id", "unknown")),
        "expected_bucket": expected_bucket,
        "actual_bucket": actual_bucket,
        "scores": {
            "correct_action": score_correct_action,
            "tool_chain_depth": score_tool_chain_depth,
            "policy_cited": score_policy_cited,
            "customer_named": score_customer_named,
            "confidence_scored": score_confidence,
        },
        "tool_call_count": tool_call_count,
        "confidence_score": confidence_score,
        "total_score": total,
    }


def evaluate_all(audit_path: str, tickets_path: str) -> dict[str, Any]:
    audit_entries = _load_json(audit_path)
    tickets = _load_json(tickets_path)

    if not isinstance(audit_entries, list):
        raise ValueError("Audit file must contain a JSON array")
    if not isinstance(tickets, list):
        raise ValueError("Tickets file must contain a JSON array")

    audit_by_ticket = {
        entry.get("ticket_id"): entry
        for entry in audit_entries
        if isinstance(entry, dict) and entry.get("ticket_id")
    }

    per_ticket: list[dict[str, Any]] = []
    total_points = 0

    for ticket in tickets:
        if not isinstance(ticket, dict):
            continue
        ticket_id = ticket.get("ticket_id")
        audit_entry = audit_by_ticket.get(ticket_id, {})
        scored = score_ticket(audit_entry, ticket)
        per_ticket.append(scored)
        total_points += scored["total_score"]

    ticket_count = len(per_ticket)
    max_points = ticket_count * 100
    pass_rate = round((total_points / max_points) * 100, 2) if max_points else 0.0
    average_score = round((total_points / ticket_count), 2) if ticket_count else 0.0

    report = {
        "summary": {
            "ticket_count": ticket_count,
            "total_points": total_points,
            "max_points": max_points,
            "pass_rate_percent": pass_rate,
            "average_score": average_score,
        },
        "weights": SCORE_WEIGHTS,
        "per_ticket": per_ticket,
    }

    output_path = Path("eval_report.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Evaluated {ticket_count} tickets")
    print(f"Average score: {average_score}/100")
    print(f"Pass rate: {pass_rate}%")
    print(f"Report written to: {output_path.resolve()}")

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scores ShopWave audit log against expected_action in tickets.json",
    )
    parser.add_argument(
        "--audit",
        default="audit_log.json",
        help="Path to audit log JSON file",
    )
    parser.add_argument(
        "--tickets",
        default="data/tickets.json",
        help="Path to source tickets JSON file",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    evaluate_all(audit_path=args.audit, tickets_path=args.tickets)


if __name__ == "__main__":
    main()
