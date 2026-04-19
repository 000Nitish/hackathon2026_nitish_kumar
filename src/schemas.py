from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


TierType = Literal["standard", "premium", "vip"]
OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]
ErrorType = Literal["timeout", "malformed", "not_found", "circuit_open", "policy_violation"]
PriorityType = Literal["low", "medium", "high", "urgent"]


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str


class CustomerResult(BaseModel):
    customer_id: str
    name: str
    email: str
    phone: str
    tier: TierType
    member_since: date
    total_orders: int = Field(ge=0)
    total_spent: float = Field(ge=0)
    address: Address
    notes: str | None = None


class OrderResult(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    quantity: int = Field(gt=0)
    status: OrderStatus
    order_date: date
    amount: float = Field(gt=0)
    delivery_date: date | None = None
    return_deadline: date | None = None
    refund_status: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_delivery_consistency(self) -> "OrderResult":
        if self.status == "delivered" and self.delivery_date is None:
            raise ValueError("Delivered order must have a delivery_date")
        return self


class ProductResult(BaseModel):
    product_id: str
    name: str
    category: str
    price: float = Field(gt=0)
    warranty_months: int = Field(ge=0)
    return_window_days: int = Field(ge=0)
    returnable: bool
    notes: str | None = None


class KBResult(BaseModel):
    query: str
    results: str = Field(min_length=1)
    sections_searched: int = Field(gt=0)


class EligibilityResult(BaseModel):
    order_id: str
    eligible: bool
    reason: str
    days_since_delivery: int | None = None
    days_remaining: int | None = None
    escalation_required: bool = False


class RefundResult(BaseModel):
    order_id: str
    transaction_id: str
    amount: float = Field(gt=0)
    timestamp: datetime


class ReplyResult(BaseModel):
    ticket_id: str
    message: str
    sent_at: datetime


class EscalateResult(BaseModel):
    escalation_id: str
    ticket_id: str
    summary: str
    priority: PriorityType
    assigned_team: str
    estimated_response_hours: int = Field(gt=0)
    created_at: datetime


class ToolError(BaseModel):
    tool_name: str
    error_type: ErrorType
    message: str
    retries_attempted: int = Field(ge=0)
    correlation_id: str


class TicketRecord(BaseModel):
    ticket_id: str
    customer_email: str
    subject: str
    body: str
    source: str
    created_at: datetime
    tier: int
    expected_action: str
