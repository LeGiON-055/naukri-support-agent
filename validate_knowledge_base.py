"""
validate_knowledge_base.py

Simple validation script for the Phase 3 knowledge base.
Checks that all 12 required files exist, are not empty, and contain 2-5 sentences.

Usage:
    python validate_knowledge_base.py
"""

import os

# The 12 required knowledge base files
REQUIRED_FILES = [
    "job_application_eligibility.txt",
    "interview_scheduling.txt",
    "offer_negotiation.txt",
    "background_verification.txt",
    "notice_period.txt",
    "referral_bonus.txt",
    "internal_transfer.txt",
    "probation_period.txt",
    "remote_work.txt",
    "diversity_hiring.txt",
    "exit_interview.txt",
    "applicant_data_retention.txt",
]

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")


def count_sentences(text):
    """Count sentences by splitting on period, question mark, or exclamation mark."""
    # Remove extra whitespace
    text = text.strip()
    if not text:
        return 0

    count = 0
    for char in text:
        if char in ".?!":
            count += 1
    return count


def validate():
    """Run all validation checks and print a report."""
    print("=" * 70)
    print("KNOWLEDGE BASE VALIDATION REPORT")
    print("=" * 70)
    print()

    all_passed = True
    results = []

    for filename in REQUIRED_FILES:
        filepath = os.path.join(KB_DIR, filename)
        exists = os.path.isfile(filepath)
        sentence_count = 0
        status = "PASS"
        issues = []

        if not exists:
            status = "FAIL"
            issues.append("File does not exist")
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                status = "FAIL"
                issues.append("File is empty")
            else:
                sentence_count = count_sentences(content)

                if sentence_count < 2:
                    status = "FAIL"
                    issues.append(f"Too few sentences ({sentence_count}, need at least 2)")
                elif sentence_count > 5:
                    status = "FAIL"
                    issues.append(f"Too many sentences ({sentence_count}, max is 5)")

        if status == "FAIL":
            all_passed = False

        results.append({
            "file": filename,
            "exists": exists,
            "sentences": sentence_count,
            "status": status,
            "issues": issues,
        })

    # Print table
    header = f"{'File':<38} {'Exists':<8} {'Sentences':<11} {'Status':<6}"
    print(header)
    print("-" * len(header))

    for r in results:
        exists_str = "Yes" if r["exists"] else "No"
        sent_str = str(r["sentences"]) if r["exists"] else "-"
        print(f"{r['file']:<38} {exists_str:<8} {sent_str:<11} {r['status']:<6}")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"  -> {issue}")

    print()
    print("=" * 70)
    if all_passed:
        print("RESULT: ALL 12 FILES PASSED VALIDATION")
    else:
        failed_count = sum(1 for r in results if r["status"] == "FAIL")
        print(f"RESULT: {failed_count} FILE(S) FAILED VALIDATION")
    print("=" * 70)


if __name__ == "__main__":
    validate()
