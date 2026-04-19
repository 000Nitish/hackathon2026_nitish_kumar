from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import structlog

from .agent import AgentResult


_audit_lock = asyncio.Lock()
_dead_letter_lock = asyncio.Lock()


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger("shopwave.audit")


class AuditLogger:
    """
    Thread-safe, concurrent-safe audit and dead-letter logging.
    """

    def __init__(
        self,
        audit_path: str = "audit_log.json",
        dead_letter_path: str = "dead_letter.json",
    ) -> None:
        self.audit_path = Path(audit_path)
        self.dead_letter_path = Path(dead_letter_path)
        self._initialize_files()

    def _initialize_files(self) -> None:
        for path in (self.audit_path, self.dead_letter_path):
            if not path.exists():
                path.write_text("[]", encoding="utf-8")
            else:
                try:
                    existing = json.loads(path.read_text(encoding="utf-8") or "[]")
                    if not isinstance(existing, list):
                        path.write_text("[]", encoding="utf-8")
                except json.JSONDecodeError:
                    path.write_text("[]", encoding="utf-8")

    async def append_result(self, result: AgentResult) -> None:
        entry: dict[str, Any] = asdict(result)
        entry["processed_at"] = datetime.now(timezone.utc).isoformat()

        async with _audit_lock:
            payload = await self._read_array_file(self.audit_path)
            payload.append(entry)
            await self._write_array_file(self.audit_path, payload)

        log.info(
            "audit_result_appended",
            ticket_id=result.ticket_id,
            correlation_id=result.correlation_id,
            status=result.status,
            final_action=result.final_action,
            tool_calls=result.tool_call_count,
        )

    async def append_dead_letter(
        self,
        ticket: dict[str, Any],
        error_chain: list[str],
        correlation_id: str,
    ) -> None:
        entry = {
            "ticket_id": ticket.get("ticket_id", "unknown"),
            "correlation_id": correlation_id,
            "customer_email": ticket.get("customer_email", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_chain": error_chain,
            "ticket": ticket,
        }

        async with _dead_letter_lock:
            payload = await self._read_array_file(self.dead_letter_path)
            payload.append(entry)
            await self._write_array_file(self.dead_letter_path, payload)

        log.info(
            "dead_letter_appended",
            ticket_id=ticket.get("ticket_id", "unknown"),
            correlation_id=correlation_id,
        )

    async def _read_array_file(self, path: Path) -> list[Any]:
        if not path.exists():
            return []
        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
            raw = await handle.read()
        if not raw.strip():
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    async def _write_array_file(self, path: Path, payload: list[Any]) -> None:
        async with aiofiles.open(path, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(payload, indent=2, ensure_ascii=True))
