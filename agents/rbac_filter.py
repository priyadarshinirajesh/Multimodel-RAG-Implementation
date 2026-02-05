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
    """Nurse view with pathology filtering"""
    
    filtered = []
    
    for e in evidence:
        report_text = e.get("report_text", "")
        nurse_text = extract_findings_only(report_text)
        
        filtered_item = {
            "patient_id": e["patient_id"],
            "modality": e["modality"],
            "organ": e.get("organ", ""),
            "report_text": nurse_text,
            "image_path": None,
            "has_image": False,
            "relevance_score": e.get("relevance_score", 0)
        }
        
        # Add filtered pathology data
        filtered_item = filter_pathology_data(filtered_item, "nurse")
        
        # Copy pathology findings from original
        if "pathology_findings" in e:
            filtered_item = filter_pathology_data(e.copy(), "nurse")
            filtered_item["image_path"] = None  # Remove image access
            filtered_item["has_image"] = False
        
        filtered.append(filtered_item)
    
    logger.info(f"[RBAC] Nurse filter applied - {len(filtered)} items")
    return filtered


def filter_for_patient(evidence: list) -> list:
    """Patient view with simplified pathology findings"""
    
    filtered = []
    
    for e in evidence:
        report_text = e.get("report_text", "")
        patient_text = simplify_for_patient(report_text)
        
        filtered_item = {
            "patient_id": e["patient_id"],
            "modality": e["modality"],
            "organ": e.get("organ", ""),
            "report_text": patient_text,
            "image_path": e.get("image_path"),
            "has_image": e.get("has_image", False),
            "relevance_score": e.get("relevance_score", 0)
        }
        
        # Add patient-friendly pathology data
        if "pathology_findings" in e or "pathology_scores" in e:
            e_copy = e.copy()
            filtered_item.update(filter_pathology_data(e_copy, "patient"))
        
        filtered.append(filtered_item)
    
    logger.info(f"[RBAC] Patient filter applied - {len(filtered)} items")
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

def filter_pathology_data(evidence: dict, role: str) -> dict:
    """
    Filter pathology detection data based on user role
    
    Args:
        evidence: Single evidence dictionary
        role: User role
        
    Returns:
        Evidence with appropriately filtered pathology data
    """
    
    if role == "doctor":
        # Doctors see everything - no filtering
        return evidence
    
    elif role == "nurse":
        # Nurses see alerts only (high-confidence findings)
        if "pathology_scores" in evidence:
            high_conf = {
                k: v for k, v in evidence["pathology_scores"].items()
                if v >= 0.7  # Only show high-confidence detections
            }
            evidence["pathology_scores"] = high_conf
            
            # Update top pathologies
            evidence["top_pathologies"] = [
                (k, v) for k, v in evidence.get("top_pathologies", [])
                if v >= 0.7
            ]
            
            # Simplify findings text
            if evidence.get("top_pathologies"):
                evidence["pathology_findings"] = (
                    "High-confidence findings: " + 
                    ", ".join([f"{k} ({v*100:.0f}%)" for k, v in evidence["top_pathologies"]])
                )
            else:
                evidence["pathology_findings"] = "No high-confidence pathologies detected."
        
        return evidence
    
    elif role == "patient":
        # Patients see simplified, layman-friendly descriptions
        if "pathology_scores" in evidence:
            # Only show findings above threshold
            significant = {
                k: v for k, v in evidence["pathology_scores"].items()
                if v >= 0.5
            }
            
            # Convert to patient-friendly language
            patient_friendly_names = {
                'Atelectasis': 'partial lung collapse',
                'Cardiomegaly': 'enlarged heart',
                'Effusion': 'fluid buildup',
                'Infiltration': 'lung tissue changes',
                'Mass': 'abnormal growth',
                'Nodule': 'small growth',
                'Pneumonia': 'lung infection',
                'Pneumothorax': 'collapsed lung',
                'Consolidation': 'dense lung tissue',
                'Edema': 'fluid accumulation',
                'Emphysema': 'lung air pockets',
                'Fibrosis': 'lung scarring',
                'Pleural_Thickening': 'thickened lung lining',
                'Hernia': 'tissue protrusion'
            }
            
            if significant:
                findings = []
                for pathology, score in sorted(significant.items(), key=lambda x: x[1], reverse=True):
                    friendly_name = patient_friendly_names.get(pathology, pathology.lower())
                    likelihood = "high" if score > 0.7 else "moderate"
                    findings.append(f"- {friendly_name.title()} detected ({likelihood} likelihood)")
                
                evidence["pathology_findings"] = "Findings from image analysis:\n" + "\n".join(findings)
                evidence["pathology_findings"] += "\n\nYour doctor will explain these findings in detail."
            else:
                evidence["pathology_findings"] = "No significant findings detected in the automated analysis."
            
            # Remove raw scores (too technical for patients)
            evidence["pathology_scores"] = {}
            evidence["top_pathologies"] = []
        
        return evidence
    
    return evidence
