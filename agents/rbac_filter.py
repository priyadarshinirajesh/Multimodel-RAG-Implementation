# agents/rbac_filter.py

from utils.logger import get_logger

logger = get_logger("RBACFilter")


def apply_rbac_filter(evidence: list, role: str) -> list:
    """
    Filters evidence based on user role.
    
    Args:
        evidence: List of evidence dictionaries
        role: One of "doctor", "nurse", "patient"
    
    Returns:
        Filtered evidence appropriate for the role
    """
    
    logger.info(f"[RBAC] Applying filter for role: {role}")
    
    if role == "doctor":
        # Doctors get full access - no filtering
        logger.info(f"[RBAC] Doctor role - returning full evidence ({len(evidence)} items)")
        return evidence
    
    elif role == "nurse":
        logger.info(f"[RBAC] Nurse role - applying care-focused filter")
        return filter_for_nurse(evidence)
    
    elif role == "patient":
        logger.info(f"[RBAC] Patient role - applying simplified filter")
        return filter_for_patient(evidence)
    
    else:
        logger.warning(f"[RBAC] Unknown role '{role}', defaulting to patient view")
        return filter_for_patient(evidence)


def filter_for_nurse(evidence: list) -> list:
    """
    Nurse view:
    - Summary of findings only (no impression/diagnosis)
    - NO image access
    - Care-oriented information
    """
    
    filtered = []
    
    for e in evidence:
        # Extract only findings section
        report_text = e.get("report_text", "")
        nurse_text = extract_findings_only(report_text)
        
        filtered.append({
            "patient_id": e["patient_id"],
            "modality": e["modality"],
            "organ": e.get("organ", ""),
            "report_text": nurse_text,
            "image_path": None,  # ❌ Nurses don't get image access
            "has_image": False,
            "relevance_score": e.get("relevance_score", 0)
        })
    
    logger.info(f"[RBAC] Nurse filter applied - {len(filtered)} items (images removed)")
    return filtered


def filter_for_patient(evidence: list) -> list:
    """
    Patient view:
    - Simplified, layman-friendly language
    - CAN view images (their own X-rays)
    - No technical jargon or diagnostic terms
    """
    
    filtered = []
    
    for e in evidence:
        report_text = e.get("report_text", "")
        patient_text = simplify_for_patient(report_text)
        
        filtered.append({
            "patient_id": e["patient_id"],
            "modality": e["modality"],
            "organ": e.get("organ", ""),
            "report_text": patient_text,
            "image_path": e.get("image_path"),  # ✅ Patients can view their images
            "has_image": e.get("has_image", False),
            "relevance_score": e.get("relevance_score", 0)
        })
    
    logger.info(f"[RBAC] Patient filter applied - {len(filtered)} items (simplified)")
    return filtered


def extract_findings_only(text: str) -> str:
    """
    Extract only the Findings section, remove Impression/Diagnosis.
    This is what nurses need for care delivery.
    """
    
    # Split by sections
    lines = text.split("\n")
    findings_section = []
    capture = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Start capturing at Findings
        if "findings:" in line_lower:
            capture = True
            continue
        
        # Stop at Impression or Diagnosis
        if capture and ("impression:" in line_lower or "diagnosis" in line_lower):
            break
        
        if capture and line.strip():
            findings_section.append(line)
    
    if findings_section:
        result = "Clinical Findings:\n" + "\n".join(findings_section)
    else:
        # Fallback if no clear structure
        result = "Clinical observations available. Refer to supervising physician for interpretation."
    
    return result.strip()


def simplify_for_patient(text: str) -> str:
    """
    Convert medical jargon to plain language.
    Remove diagnostic sections entirely.
    """
    
    # Dictionary of medical terms → plain language
    replacements = {
        "pleural effusion": "fluid around the lungs",
        "cardiomegaly": "enlarged heart",
        "consolidation": "area of concern in lung tissue",
        "opacity": "cloudy area",
        "infiltrate": "abnormal appearance",
        "mediastinum": "central chest area",
        "costophrenic angle": "lower chest area",
        "hemidiaphragm": "breathing muscle",
        "pulmonary": "lung",
        "cardiac silhouette": "heart outline",
        "no acute": "no immediate concern",
        "unremarkable": "normal",
        "within normal limits": "normal"
    }
    
    simplified = text
    
    # Apply replacements
    for medical_term, plain_term in replacements.items():
        simplified = simplified.replace(medical_term, plain_term)
        # Also handle capitalized versions
        simplified = simplified.replace(medical_term.title(), plain_term.title())
    
    # Remove Impression section (too technical for patients)
    if "Impression:" in simplified:
        simplified = simplified.split("Impression:")[0]
    
    # Add patient-friendly prefix
    simplified = "Your Imaging Report (Simplified):\n\n" + simplified
    
    return simplified.strip()
