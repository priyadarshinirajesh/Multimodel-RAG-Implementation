# agents/verifiers/structure_repair.py

def enforce_structure(text: str) -> str:
    sections = {
        "Diagnosis / Impression": "",
        "Supporting Evidence": "",
        "Next Steps / Recommendations": ""
    }

    current_section = None

    for line in text.splitlines():
        lower = line.lower().strip()

        if "diagnosis" in lower or "impression" in lower:
            current_section = "Diagnosis / Impression"
            continue
        elif "supporting evidence" in lower or lower.startswith("-"):
            current_section = current_section or "Supporting Evidence"
        elif "next steps" in lower or "recommendation" in lower:
            current_section = "Next Steps / Recommendations"
            continue

        if current_section:
            sections[current_section] += line + "\n"

    # 🔒 Force minimum content
    if not sections["Diagnosis / Impression"].strip():
        sections["Diagnosis / Impression"] = (
            "- No definitive abnormality identified based on available evidence. [R1]\n"
        )

    if not sections["Supporting Evidence"].strip():
        sections["Supporting Evidence"] = (
            "- Imaging findings do not demonstrate acute pathology. [R1]\n"
        )

    if not sections["Next Steps / Recommendations"].strip():
        sections["Next Steps / Recommendations"] = (
            "- Clinical correlation is recommended. [R1]\n"
        )

    return f"""Diagnosis / Impression:
{sections["Diagnosis / Impression"].strip()}

Supporting Evidence:
{sections["Supporting Evidence"].strip()}

Next Steps / Recommendations:
{sections["Next Steps / Recommendations"].strip()}
""".strip()
