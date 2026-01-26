# agents/verifiers/evidence_consistency_checker.py

from utils.logger import get_logger

logger = get_logger("EvidenceConsistencyChecker")


def check_evidence_consistency(response: str, evidence: list) -> dict:
    issues = []

    if len(evidence) >= 2:
        citations = set()
        for i in range(1, len(evidence) + 1):
            if f"[R{i}]" in response:
                citations.add(i)

        if len(citations) < 2 and "conflicting findings" not in response.lower():
            issues.append("Insufficient citation coverage")

    response_says_no_effusion = "no pleural effusion" in response.lower()

    report_mentions_effusion = any(
        "effusion" in e.get("report_text", "").lower()
        for e in evidence
    )

    if response_says_no_effusion and report_mentions_effusion:
        issues.append("Effusion contradiction across reports")

    if issues:
        logger.warning(f"[ConsistencyChecker] Issues detected: {issues}")
        return {"passed": False, "issues": issues}

    return {"passed": True, "issues": []}
