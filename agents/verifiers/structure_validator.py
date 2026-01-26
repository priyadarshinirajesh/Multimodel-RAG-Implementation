REQUIRED_SECTIONS = [
    "Diagnosis / Impression:",
    "Supporting Evidence:",
    "Next Steps / Recommendations:"
]

def validate_structure(text: str) -> dict:
    missing = [
        section
        for section in REQUIRED_SECTIONS
        if section.lower() not in text.lower()
    ]

    return {
        "passed": len(missing) == 0,
        "missing_sections": missing
    }
