"""
dataset.py — Deterministic Job Application Dataset Generator

Generates at least 40 synthetic job application records with fields:
- record_id, category, status, expected_salary_inr,
  days_since_created, flagged_priority_review

Uses a fixed random seed so every run produces identical output.
"""

import random

# ── Fixed seed for deterministic output ──────────────────────────
SEED = 42
random.seed(SEED)

# ── Required categories and statuses ─────────────────────────────
CATEGORIES = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
]

STATUSES = [
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected",
]

# Realistic salary ranges per category (min, max in INR)
SALARY_RANGES = {
    "Software Engineer": (600000, 2500000),
    "Data Analyst":      (400000, 1800000),
    "Product Manager":   (800000, 3000000),
    "HR Executive":      (300000, 1200000),
    "Sales Associate":   (250000, 1000000),
}

TOTAL_RECORDS = 45
PRIORITY_PROBABILITY = 0.18  # ~18% chance → naturally lands in 10-30%


# ── Record generator ─────────────────────────────────────────────
def generate_record(record_id, category, status):
    """Generate a single job application record."""
    salary_min, salary_max = SALARY_RANGES[category]
    expected_salary = random.randint(salary_min, salary_max)
    # Round to nearest ₹10,000 for realism
    expected_salary = round(expected_salary, -4)

    days_since_created = random.randint(0, 30)
    flagged_priority_review = random.random() < PRIORITY_PROBABILITY

    return {
        "record_id": record_id,
        "category": category,
        "status": status,
        "expected_salary_inr": expected_salary,
        "days_since_created": days_since_created,
        "flagged_priority_review": flagged_priority_review,
    }


# ── Dataset generator ────────────────────────────────────────────
def generate_dataset():
    """
    Generate the full dataset with coverage guarantees.

    Strategy:
      Phase 1 — 3 rounds × 5 categories = 15 records.
                First round also maps each category to a unique status
                so all 5 statuses are guaranteed to appear.
      Phase 2 — Fill remaining records with random category/status.
    """
    records = []
    record_id = 1

    # Phase 1: guarantee every category ≥ 3 and every status ≥ 1
    for round_num in range(3):
        for i, category in enumerate(CATEGORIES):
            if round_num == 0:
                # First round: pair each category with a unique status
                status = STATUSES[i]
            else:
                status = random.choice(STATUSES)

            record = generate_record(f"APP-{record_id:03d}", category, status)
            records.append(record)
            record_id += 1

    # Phase 2: fill the rest randomly
    remaining = TOTAL_RECORDS - len(records)
    for _ in range(remaining):
        category = random.choice(CATEGORIES)
        status = random.choice(STATUSES)
        record = generate_record(f"APP-{record_id:03d}", category, status)
        records.append(record)
        record_id += 1

    return records


# ── Statistics ────────────────────────────────────────────────────
def print_statistics(records):
    """Print useful dataset statistics and return counts for verification."""
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal Records: {len(records)}")

    # Category distribution
    print("\n--- Category Distribution ---")
    category_counts = {}
    for record in records:
        cat = record["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat in CATEGORIES:
        print(f"  {cat}: {category_counts.get(cat, 0)}")

    # Status distribution
    print("\n--- Status Distribution ---")
    status_counts = {}
    for record in records:
        st = record["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
    for st in STATUSES:
        print(f"  {st}: {status_counts.get(st, 0)}")

    # Priority review
    priority_count = sum(1 for r in records if r["flagged_priority_review"])
    priority_pct = (priority_count / len(records)) * 100
    print(f"\n--- Priority Review ---")
    print(f"  Flagged: {priority_count}/{len(records)} ({priority_pct:.1f}%)")

    # Salary range
    salaries = [r["expected_salary_inr"] for r in records]
    print(f"\n--- Salary Range (INR) ---")
    print(f"  Min: {min(salaries):,}")
    print(f"  Max: {max(salaries):,}")
    print(f"  Avg: {sum(salaries) // len(salaries):,}")

    # Days range
    days = [r["days_since_created"] for r in records]
    print(f"\n--- Days Since Created ---")
    print(f"  Min: {min(days)}")
    print(f"  Max: {max(days)}")
    print(f"  Avg: {sum(days) / len(days):.1f}")

    print("\n" + "=" * 60)

    return category_counts, status_counts, priority_pct


# ── Requirements verification ────────────────────────────────────
def verify_requirements(records, category_counts, status_counts, priority_pct):
    """Automatically verify every capstone requirement."""
    print("\nREQUIREMENTS VERIFICATION")
    print("=" * 60)

    required_fields = {
        "record_id", "category", "status", "expected_salary_inr",
        "days_since_created", "flagged_priority_review",
    }

    checks = [
        (
            "At least 40 records",
            len(records) >= 40,
            f"{len(records)} records",
        ),
        (
            "All 5 categories present",
            all(cat in category_counts for cat in CATEGORIES),
            "",
        ),
        (
            "Every category >= 3 times",
            min(category_counts.get(c, 0) for c in CATEGORIES) >= 3,
            f"min = {min(category_counts.get(c, 0) for c in CATEGORIES)}",
        ),
        (
            "All 5 statuses present",
            all(st in status_counts for st in STATUSES),
            "",
        ),
        (
            "Every status >= 1 time",
            min(status_counts.get(s, 0) for s in STATUSES) >= 1,
            f"min = {min(status_counts.get(s, 0) for s in STATUSES)}",
        ),
        (
            "All required fields present",
            all(required_fields.issubset(r.keys()) for r in records),
            "",
        ),
        (
            "Salaries are numeric",
            all(isinstance(r["expected_salary_inr"], (int, float)) for r in records),
            "",
        ),
        (
            "days_since_created in 0-30",
            all(0 <= r["days_since_created"] <= 30 for r in records),
            "",
        ),
        (
            "flagged_priority_review is bool",
            all(isinstance(r["flagged_priority_review"], bool) for r in records),
            "",
        ),
        (
            "Priority % in 10-30%",
            10 <= priority_pct <= 30,
            f"{priority_pct:.1f}%",
        ),
    ]

    all_passed = True
    for name, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        detail_str = f" ({detail})" if detail else ""
        print(f"  [{icon}] {name}{detail_str}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL REQUIREMENTS PASSED!")
    else:
        print("SOME REQUIREMENTS FAILED - review above.")
    print("=" * 60)

    return all_passed


# ── Main execution ───────────────────────────────────────────────
JOB_APPLICATIONS = generate_dataset()

# Show sample records
print("--- Sample Records (first 5) ---\n")
for record in JOB_APPLICATIONS[:5]:
    print(record)

# Print statistics and verify
cat_counts, stat_counts, prio_pct = print_statistics(JOB_APPLICATIONS)
verify_requirements(JOB_APPLICATIONS, cat_counts, stat_counts, prio_pct)
