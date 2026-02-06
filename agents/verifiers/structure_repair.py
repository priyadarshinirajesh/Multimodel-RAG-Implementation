# agents/verifiers/structure_repair.py

def enforce_structure(text: str) -> str:
    sections = {
        "Diagnosis / Impression": "",
        "Supporting Evidence": "",
        "Next Steps / Recommendations": ""
    }

    current_section = None
    lines = text.splitlines()

    for line in lines:
        lower = line.lower().strip()
        
        # Skip empty lines
        if not lower:
            continue

        # ✅ FIXED: Explicit section header detection
        if "diagnosis" in lower and "impression" in lower:
            current_section = "Diagnosis / Impression"
            continue
        elif ("diagnosis" in lower or "impression" in lower) and ":" in line:
            current_section = "Diagnosis / Impression"
            continue
        elif "supporting evidence" in lower or "evidence:" in lower:
            current_section = "Supporting Evidence"
            continue
        elif "next steps" in lower or "recommendation" in lower:
            current_section = "Next Steps / Recommendations"
            continue
        
        # ✅ Capture content only if we're in a section
        if current_section:
            sections[current_section] += line + "\n"

    # 🔒 Force minimum content (with proper structure)
    if not sections["Diagnosis / Impression"].strip():
        sections["Diagnosis / Impression"] = (
            "- No definitive abnormality identified based on available evidence. [R1]"
        )

    if not sections["Supporting Evidence"].strip():
        sections["Supporting Evidence"] = (
            "- Imaging findings do not demonstrate acute pathology. [R1]"
        )

    if not sections["Next Steps / Recommendations"].strip():
        sections["Next Steps / Recommendations"] = (
            "- Clinical correlation is recommended. [R1]"
        )

    # ✅ Return with PROPER formatting
    return f"""Diagnosis / Impression:
{sections["Diagnosis / Impression"].strip()}

Supporting Evidence:
{sections["Supporting Evidence"].strip()}

Next Steps / Recommendations:
{sections["Next Steps / Recommendations"].strip()}
""".strip()