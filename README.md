# ShopWave - Autonomous Support Resolution Agent

ShopWave is a production-oriented autonomous support agent that ingests tickets, runs a ReAct loop with tool use, applies policy-grounded reasoning, and produces audit-grade telemetry for every decision.

## Architecture Diagram

```text
+-----------------------------------------------------------------+
|                     main.py - Orchestrator                      |
|  +--------------+    asyncio.gather()    +------------------+   |
|  |TicketIngester| ---- concurrent ----->  |  AgentWorker[N]  |   |
|  |  (tickets.   |     (max 5 parallel)   |  (one per ticket)|   |
|  |   json)      |                        +--------+---------+   |
|  +--------------+                                 |             |
|                                                   v             |
|                              +-----------------------------+   |
|                              |     agent.py - ReAct Engine |   |
|                              |  Thought -> Action -> Observe|   |
|                              |  (max 10 iterations/ticket) |   |
|                              +------------+----------------+   |
|                                           |                     |
|              +----------------------------+--------------+     |
|              v                            v              v     |
|         tools.py                    Gemini API      AuditLogger |
|    (mock data layer with       (google-genai SDK)  (audit_log   |
|     realistic failures)         Function Calling    .json +      |
|                                                    dead_letter   |
|                                                    .json)        |
+-----------------------------------------------------------------+
```

## Quick Start

```bash
git clone <your-repo-url> shopwave-agent && cd shopwave-agent
cp .env.example .env && edit .env  # add GOOGLE_API_KEY
python main.py
```

## Docker Instructions

```bash
cp .env.example .env  # add GOOGLE_API_KEY
docker compose build
docker compose up
```

## Evaluation Criteria Mapping

The five non-negotiable constraints are satisfied as follows:

1. Deep tool chaining (min 3 verification tools before terminal action)
   - Implemented in `run_agent` system guard and block injection before `send_reply`/`escalate`.
   - References: `src/agent.py:44`, `src/agent.py:45`, `src/agent.py:613`.
2. True concurrency with failure isolation
   - Batch dispatch uses `asyncio.gather(..., return_exceptions=True)` plus `Throttler`.
   - References: `main.py:166`, `main.py:191`, `main.py:193`.
3. Tiered resilience (retry + circuit breaker + dead-letter)
   - Retry/backoff: tenacity wrappers in `src/tools.py`.
   - Circuit breaker: `check_refund_eligibility` circuit-open handling in `src/tools.py`.
   - Dead-letter: policy-violation + unrecoverable paths in `src/agent.py`, plus logger support in `src/logger.py`.
   - References: `src/tools.py:278`, `src/tools.py:317`, `src/tools.py:553`, `src/tools.py:562`, `src/agent.py:736`, `src/logger.py:74`.
4. Full explainability
   - Reasoning trace captures `Thought:` lines and explicit state transitions per tool.
   - References: `src/agent.py:384`, `src/agent.py:464`, `src/agent.py:571`.
5. Schema validation before action
   - All tool I/O contracts are Pydantic v2 models.
   - Tool outputs are validated in `src/tools.py` and normalized in `execute_tool_safely`.
   - References: `src/schemas.py:35`, `src/schemas.py:104`, `src/tools.py:334`, `src/tools.py:355`, `src/tools.py:471`, `src/agent.py:788`.

## Sample Audit Log Entry

```json
{
  "ticket_id": "TKT-001",
  "correlation_id": "9c1f2b4e",
  "customer_email": "alice.turner@email.com",
  "status": "resolved",
  "final_action": "refund_issued",
  "reasoning_trace": [
    "Thought: [STATE: INGESTING] [POLICY: System State Machine] Ticket ingested; starting verification workflow.",
    "Thought: [STATE: CUSTOMER_VERIFIED] [POLICY: Customer Tiers & Privileges] Verified customer tier and notes."
  ],
  "tools_called": [
    {
      "tool": "get_customer",
      "args": {"email": "alice.turner@email.com", "correlation_id": "9c1f2b4e"},
      "success": true,
      "retries": 0
    }
  ],
  "confidence_score": 0.88,
  "processing_time_ms": 2140,
  "tool_call_count": 4,
  "retry_count": 1,
  "error": null,
  "customer_name": "Alice Turner",
  "order_id": "ORD-1001",
  "processed_at": "2026-04-18T15:24:10.112345+00:00"
}
```

Field guide:
- `correlation_id`: trace token shared across tools/logs for one ticket.
- `reasoning_trace`: explainability ledger (model thought + state machine transitions).
- `tools_called`: execution evidence with arguments, retries, and failures.
- `retry_count`: total retry attempts consumed by resilient tools.
- `final_action` + `status`: explicit terminal outcome for scoring and ops routing.

## ReAct Worked Example (TKT-001)

```text
Thought: [STATE: INGESTING] [POLICY: Customer Tiers & Privileges] I must verify tier before any decision.
Action: get_customer(email="alice.turner@email.com")
Observation: tier=vip, customer_id=C001, notes=Long-standing VIP.

Thought: [STATE: CUSTOMER_VERIFIED] [POLICY: Return Policy - Standard Return Window] Need order facts and delivery timeline.
Action: get_order(order_id="ORD-1001")
Observation: status=delivered, product_id=P001, amount=129.99, delivery_date=2024-02-14

Thought: [STATE: ORDER_VERIFIED] [POLICY: Return Policy - Damaged or Defective on Arrival] Defect claim can override normal return window, but product context still needed.
Action: get_product(product_id="P001")
Observation: category=electronics, warranty_months=12, return_window_days=30

Thought: [STATE: POLICY_CHECKED] [POLICY: Refund Policy - Refund Eligibility] Must verify refund eligibility before irreversible action.
Action: check_refund_eligibility(order_id="ORD-1001")
Observation: eligible=true, escalation_required=false, reason="Within return window (X days remaining)"

Thought: [STATE: ELIGIBILITY_VERIFIED] [POLICY: Refund Policy - Refund Eligibility] Confidence is high and amount is below escalation threshold; proceed with refund.
Action: issue_refund(order_id="ORD-1001", amount=129.99)
Observation: transaction_id=..., timestamp=...

Thought: [STATE: ACTION_TAKEN] [POLICY: Communication Guidelines] Send empathetic, explicit confirmation with next steps.
Action: send_reply(ticket_id="TKT-001", message="Alice, I have approved your refund ...")
Observation: reply_sent=true

Thought: [STATE: ACTION_TAKEN] [POLICY: Confidence Scoring] confidence_score: 0.88 - complete data, clear policy, verified customer.
Final: resolved / refund_issued
```

## Known Limitations

1. Knowledge-base search is keyword scoring (no embeddings/vector retrieval).
2. Multi-order disambiguation is still model-led and can be improved with a dedicated lookup tool.
3. No external fraud service integration yet; fraud detection relies on prompt-guided reasoning and available data.
4. Audit/dead-letter files are JSON arrays rewritten on append; this is safe with locks but not optimized for very large runs.
5. The evaluation scorer uses heuristic expected-action bucketing from free-text labels.

## Running the Evaluator

```bash
python tests/eval_agent.py --audit audit_log.json --tickets data/tickets.json
```

Interpretation:
- `average_score`: mean ticket score out of 100.
- `pass_rate_percent`: total points divided by theoretical maximum.
- `eval_report.json`: includes criterion-level per-ticket breakdown for debugging misses.
