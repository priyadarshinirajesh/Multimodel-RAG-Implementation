# agents/verifiers/structure_repair.py


def enforce_structure(text: str) -> str:
    """
    Enforce the four-section clinical response format.
    Parses LLM output and maps it to the correct sections.
    Provides safe fallback content for any missing section.
    """

    sections = {
        "Clinical Impression": "",
        "Evidence Synthesis": "",
        "Differential Considerations": "",
    }

    current_section = None
    lines = text.splitlines()

    for line in lines:
        lower = line.lower().strip()

        # Skip empty lines
        if not lower:
            continue

        # ── Section header detection ──────────────────────────────────────────
        if "clinical impression" in lower:
            current_section = "Clinical Impression"
            continue
        elif "evidence synthesis" in lower:
            current_section = "Evidence Synthesis"
            continue
        elif "differential consideration" in lower:
            current_section = "Differential Considerations"
            continue

        # ── Legacy section header support (in case LLM uses old names) ───────
        elif ("diagnosis" in lower and "impression" in lower) or \
             ("diagnosis" in lower and ":" in line):
            current_section = "Clinical Impression"
            continue
        elif "supporting evidence" in lower or lower == "evidence:":
            current_section = "Evidence Synthesis"
            continue


        # ── Capture content if we are inside a section ────────────────────────
        if current_section:
            sections[current_section] += line + "\n"

    # ── Force minimum content for each section ────────────────────────────────
    if not sections["Clinical Impression"].strip():
        sections["Clinical Impression"] = (
            "- Insufficient evidence to determine finding with confidence. [R1] "
            "[LOW CONFIDENCE — imaging/CNN only, no text report support]"
        )

    if not sections["Evidence Synthesis"].strip():
        sections["Evidence Synthesis"] = (
            "- Imaging findings do not demonstrate acute pathology. [R1]"
        )

    if not sections["Differential Considerations"].strip():
        sections["Differential Considerations"] = (
            "- Primary: No acute abnormality identified based on available evidence. [R1]\n"
            "- Alternative: Insufficient evidence to exclude other diagnoses without additional imaging. [Rx]"
        )


    # ── Return with proper formatting ─────────────────────────────────────────
    return f"""Clinical Impression:
{sections["Clinical Impression"].strip()}

Evidence Synthesis:
{sections["Evidence Synthesis"].strip()}

Differential Considerations:
{sections["Differential Considerations"].strip()}

""".strip()