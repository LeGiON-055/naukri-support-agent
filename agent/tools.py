"""
tools.py — Domain Tools for Naukri Domain Support Agent (Phase 8 / Task 6)

Implements domain-specific tools called by the support agent:
1. check_job_application_status(record_id)
   - Looks up job application details in dataset.JOB_APPLICATIONS
   - Returns status, expected salary, and computed escalation score
   - Applies an empirical threshold calibrated to the dataset's 80th percentile
2. calculate_escalation_score(record)
   - Combines flagged_priority_review (50%) and normalized recency (50%)
   - Bounded strictly within [0.0, 1.0]
"""

import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import JOB_APPLICATIONS

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

MAX_DAYS_WINDOW = 30.0    # Lifetime bounds for days_since_created (0-30 days)
WEIGHT_PRIORITY = 0.50    # Priority review flag weight
WEIGHT_RECENCY = 0.50     # Waiting latency weight

# Calibrated threshold based on the 80th percentile of dataset (P80 = 0.4667)
# Setting to 0.45 escalates top 22.2% (10/45 records): all 8 priority-flagged
# records + top 2 longest-waiting unflagged records (28 days).
ESCALATION_THRESHOLD = 0.45

# O(1) in-memory lookup index
_APPLICATION_INDEX = {record["record_id"]: record for record in JOB_APPLICATIONS}


# ---------------------------------------------------------------------------
# Escalation Score Calculation
# ---------------------------------------------------------------------------

def calculate_escalation_score(record):
    """
    Calculate the escalation score for a job application record.

    Formula:
        escalation_score = 0.50 * priority_flag + 0.50 * (days_since_created / 30.0)

    Where:
        priority_flag = 1.0 if flagged_priority_review is True else 0.0
        days_since_created is clamped to [0, 30] before normalization

    Bounds:
        Min possible = 0.50*(0) + 0.50*(0/30) = 0.0
        Max possible = 0.50*(1) + 0.50*(30/30) = 1.0
        The output is strictly bounded within [0.0, 1.0].

    Args:
        record (dict): Application record dictionary.

    Returns:
        float: Escalation score rounded to 4 decimal places.
    """
    priority_flag = 1.0 if record.get("flagged_priority_review", False) else 0.0
    raw_days = record.get("days_since_created", 0)
    clamped_days = max(0.0, min(float(raw_days), MAX_DAYS_WINDOW))
    recency_norm = clamped_days / MAX_DAYS_WINDOW

    score = (WEIGHT_PRIORITY * priority_flag) + (WEIGHT_RECENCY * recency_norm)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Core Tool Function
# ---------------------------------------------------------------------------

def check_job_application_status(record_id):
    """
    Look up a job application by record_id and evaluate escalation status.

    Args:
        record_id (str): The unique application identifier (e.g. 'APP-001').

    Returns:
        dict: Structured result containing:
            - found (bool): True if record exists, False otherwise
            - record_id (str): Requested identifier
            - category (str | None): Job category (e.g. 'Software Engineer')
            - status (str | None): Application status (e.g. 'Applied', 'Screening')
            - expected_salary_inr (int | None): Salary expectation in INR
            - days_since_created (int | None): Days elapsed since submission
            - flagged_priority_review (bool | None): Whether HR flagged priority
            - escalation_score (float | None): Urgency score between 0.0 and 1.0
            - escalated (bool): True if escalation_score >= ESCALATION_THRESHOLD
            - escalation_threshold (float): Active threshold (0.45)
            - message (str): Clear human-readable summary of status and escalation
    """
    clean_id = str(record_id).strip() if record_id else ""
    record = _APPLICATION_INDEX.get(clean_id)

    if not record:
        return {
            "found": False,
            "record_id": clean_id,
            "category": None,
            "status": None,
            "expected_salary_inr": None,
            "days_since_created": None,
            "flagged_priority_review": None,
            "escalation_score": None,
            "escalated": False,
            "escalation_threshold": ESCALATION_THRESHOLD,
            "message": (
                f"Job application record '{clean_id}' was not found in the database. "
                f"Please verify the ID format (e.g., 'APP-001')."
            ),
        }

    score = calculate_escalation_score(record)
    is_escalated = score >= ESCALATION_THRESHOLD

    status_str = record["status"]
    salary_str = f"INR {record['expected_salary_inr']:,}"
    days = record["days_since_created"]
    priority = record["flagged_priority_review"]

    if is_escalated:
        escalation_reason = []
        if priority:
            escalation_reason.append("marked for Priority Review")
        if days >= 20:
            escalation_reason.append(f"pending for {days} days")
        reason_desc = " and ".join(escalation_reason) if escalation_reason else f"escalation score ({score:.4f})"
        msg = (
            f"Application '{clean_id}' for {record['category']} is currently in '{status_str}' status "
            f"with expected salary of {salary_str}. "
            f"This case has been ESCALATED (score: {score:.4f} >= {ESCALATION_THRESHOLD}) "
            f"because it is {reason_desc}. A senior recruiter will expedite this review."
        )
    else:
        msg = (
            f"Application '{clean_id}' for {record['category']} is currently in '{status_str}' status "
            f"with expected salary of {salary_str}. "
            f"It was created {days} days ago and is progressing through standard evaluation "
            f"(escalation score: {score:.4f} < {ESCALATION_THRESHOLD})."
        )

    return {
        "found": True,
        "record_id": clean_id,
        "category": record["category"],
        "status": status_str,
        "expected_salary_inr": record["expected_salary_inr"],
        "days_since_created": days,
        "flagged_priority_review": priority,
        "escalation_score": score,
        "escalated": is_escalated,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "message": msg,
    }


def get_all_records():
    """Return a shallow copy of all application records."""
    return list(JOB_APPLICATIONS)
