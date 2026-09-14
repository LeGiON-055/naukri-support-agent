"""
eval_set.py — Evaluation Question Set for Phase 7

Contains 12 carefully designed evaluation queries covering all 12 Knowledge Base
policy topics. Each query has explicitly defined expected parent document(s).

Used to benchmark retrieval performance between:
1. Fixed-size-with-overlap chunking
2. Sentence-based chunking
"""

EVALUATION_SET = [
    {
        "id": "EVAL-01",
        "topic": "job_application_eligibility",
        "query": "What are the minimum qualifications and eligibility requirements to apply for a job posting?",
        "expected_documents": ["job_application_eligibility.txt"],
    },
    {
        "id": "EVAL-02",
        "topic": "interview_scheduling",
        "query": "How can a candidate reschedule an interview and what is the notice window required?",
        "expected_documents": ["interview_scheduling.txt"],
    },
    {
        "id": "EVAL-03",
        "topic": "offer_negotiation",
        "query": "Can candidates negotiate salary and benefits after receiving an initial offer?",
        "expected_documents": ["offer_negotiation.txt"],
    },
    {
        "id": "EVAL-04",
        "topic": "probation_period",
        "query": "How long is the employee probation period and how is performance evaluated?",
        "expected_documents": ["probation_period.txt"],
    },
    {
        "id": "EVAL-05",
        "topic": "remote_work",
        "query": "What is the policy and manager approval process for remote work eligibility?",
        "expected_documents": ["remote_work.txt"],
    },
    {
        "id": "EVAL-06",
        "topic": "applicant_data_retention",
        "query": "How long does the company retain applicant personal data and resume records?",
        "expected_documents": ["applicant_data_retention.txt"],
    },
    {
        "id": "EVAL-07",
        "topic": "referral_bonus",
        "query": "When is an employee referral bonus paid out and what are the eligibility terms?",
        "expected_documents": ["referral_bonus.txt"],
    },
    {
        "id": "EVAL-08",
        "topic": "background_verification",
        "query": "What background verification checks are performed on education and employment history?",
        "expected_documents": ["background_verification.txt"],
    },
    {
        "id": "EVAL-09",
        "topic": "internal_transfer",
        "query": "When can an employee apply for an internal job transfer to another team?",
        "expected_documents": ["internal_transfer.txt"],
    },
    {
        "id": "EVAL-10",
        "topic": "diversity_hiring",
        "query": "What are the company's equal opportunity and diversity hiring principles?",
        "expected_documents": ["diversity_hiring.txt"],
    },
    {
        "id": "EVAL-11",
        "topic": "notice_period",
        "query": "What are the rules regarding notice period disclosure and agreeing on joining dates?",
        "expected_documents": ["notice_period.txt"],
    },
    {
        "id": "EVAL-12",
        "topic": "exit_interview",
        "query": "What is the purpose of an exit interview and is participation mandatory?",
        "expected_documents": ["exit_interview.txt"],
    },
]
