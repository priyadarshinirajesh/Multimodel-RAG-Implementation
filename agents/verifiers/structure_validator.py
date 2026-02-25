import re

def validate_structure(text: str) -> dict:
    patterns = [
        r"(?im)^Diagnosis / Impression:\s*$",
        r"(?im)^Supporting Evidence:\s*$",
        r"(?im)^Next Steps / Recommendations:\s*$"
    ]
    missing = [p for p in patterns if not re.search(p, text)]

    return {
        "passed": len(missing) == 0,
        "missing_sections": missing
    }
