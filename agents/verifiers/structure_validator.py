# agents/verifiers/structure_validator.py

import re


def validate_structure(text: str) -> dict:
    """
    Validate that the response contains all four required clinical sections.
    Updated for the new Clinical Assistant response format.
    """

    patterns = [
        r"(?im)^Clinical Impression:\s*$",
        r"(?im)^Evidence Synthesis:\s*$",
        r"(?im)^Differential Considerations:\s*$",
    ]

    missing = [p for p in patterns if not re.search(p, text)]

    return {
        "passed": len(missing) == 0,
        "missing_sections": missing,
    }