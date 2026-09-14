"""
schemas.py — Structured Output Schemas & Response Standardization (Phase 9 / Task 9)

Defines the unified, strict Pydantic JSON Schema for every agent response.
Every response produced by the Naukri Domain Support Agent conforms to this model.

Key Components:
- IntentType: Literal restricting intents to 4 valid domain categories.
- ApplicationDetails: Nested telemetry for job application status lookups.
- RefusalInfo: Nested metadata for fallbacks and safety refusals.
- AgentResponse: Root response contract.
- validate_agent_response(data): In-code validation helper returning (is_valid, model, error).
"""

from typing import Literal, Optional, List, Tuple, Union, Any, Dict
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Allowed Intent Literals
# ---------------------------------------------------------------------------

IntentType = Literal[
    "policy_inquiry",
    "application_status",
    "out_of_scope",
    "security_violation",
]

ALLOWED_INTENTS = [
    "policy_inquiry",
    "application_status",
    "out_of_scope",
    "security_violation",
]


# ---------------------------------------------------------------------------
# Nested Models
# ---------------------------------------------------------------------------

class ApplicationDetails(BaseModel):
    """Structured details for a job application status query."""
    record_id: str = Field(
        description="Unique application identifier, e.g. 'APP-001'"
    )
    category: Optional[str] = Field(
        default=None,
        description="Job role category, e.g. 'Software Engineer'"
    )
    status: Optional[str] = Field(
        default=None,
        description="Application status, e.g. 'Screening', 'Applied'"
    )
    expected_salary_inr: Optional[int] = Field(
        default=None,
        ge=0,
        description="Expected salary in INR"
    )
    days_since_created: Optional[int] = Field(
        default=None,
        ge=0,
        description="Days elapsed since application was created"
    )
    flagged_priority_review: Optional[bool] = Field(
        default=None,
        description="Whether HR marked this application for priority review"
    )
    escalation_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Computed urgency score strictly between 0.0 and 1.0"
    )
    escalation_threshold: float = Field(
        default=0.45,
        description="Active escalation threshold calibrated from dataset P80"
    )


class RefusalInfo(BaseModel):
    """Metadata explaining why a query was rejected or refused."""
    reason: str = Field(
        description="Standard reason identifier, e.g. 'retrieval_threshold', 'prompt_injection'"
    )
    details: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the refusal reason"
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Observed cosine similarity if threshold-rejected"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="The active threshold compared against"
    )


# ---------------------------------------------------------------------------
# Root Response Model
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """
    Standardized, validated response contract for the Naukri Domain Support Agent.
    Every agent invocation produces a response conforming to this model.
    """
    intent: IntentType = Field(
        description="Classified conversation intent"
    )
    answer: str = Field(
        min_length=1,
        description="Natural language response text presented to the user"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of parent policy documents referenced (citations)"
    )
    is_escalated: bool = Field(
        default=False,
        description="Whether this query triggered human recruiter escalation"
    )
    application_details: Optional[ApplicationDetails] = Field(
        default=None,
        description="Structured application telemetry if status query"
    )
    refusal_info: Optional[RefusalInfo] = Field(
        default=None,
        description="Refusal telemetry if query rejected"
    )


# ---------------------------------------------------------------------------
# In-Code Validation Function
# ---------------------------------------------------------------------------

def validate_agent_response(data: Union[Dict[str, Any], AgentResponse]) -> Tuple[bool, Optional[AgentResponse], Optional[str]]:
    """
    Validate a response payload against the AgentResponse schema.

    Args:
        data (dict | AgentResponse): Payload to validate.

    Returns:
        tuple: (is_valid: bool, validated_model: AgentResponse | None, error_message: str | None)
    """
    if isinstance(data, AgentResponse):
        return True, data, None

    if not isinstance(data, dict):
        return False, None, f"Expected dictionary or AgentResponse instance, got {type(data).__name__}"

    try:
        validated = AgentResponse.model_validate(data)
        return True, validated, None
    except ValidationError as e:
        return False, None, str(e)


# ---------------------------------------------------------------------------
# Response Builder Helpers
# ---------------------------------------------------------------------------

def build_policy_response(answer: str, sources: Optional[List[str]] = None) -> AgentResponse:
    """Convenience builder for in-scope policy responses."""
    return AgentResponse(
        intent="policy_inquiry",
        answer=answer,
        sources=sources or [],
        is_escalated=False,
        application_details=None,
        refusal_info=None,
    )


def build_status_response(
    answer: str,
    application_details: Union[Dict[str, Any], ApplicationDetails],
    is_escalated: bool = False
) -> AgentResponse:
    """Convenience builder for job application status responses."""
    details = (
        application_details
        if isinstance(application_details, ApplicationDetails)
        else ApplicationDetails.model_validate(application_details)
    )
    return AgentResponse(
        intent="application_status",
        answer=answer,
        sources=[],
        is_escalated=is_escalated,
        application_details=details,
        refusal_info=None,
    )


def build_refusal_response(
    intent: Literal["out_of_scope", "security_violation"],
    answer: str,
    reason: str,
    details: Optional[str] = None,
    similarity_score: Optional[float] = None,
    threshold: Optional[float] = None,
) -> AgentResponse:
    """Convenience builder for fallback and security refusal responses."""
    refusal = RefusalInfo(
        reason=reason,
        details=details,
        similarity_score=similarity_score,
        threshold=threshold,
    )
    return AgentResponse(
        intent=intent,
        answer=answer,
        sources=[],
        is_escalated=False,
        application_details=None,
        refusal_info=refusal,
    )
