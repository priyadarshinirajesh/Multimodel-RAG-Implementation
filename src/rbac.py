# src/rbac.py

def apply_rbac(role, retrieved_rows):
    """
    retrieved_rows = list of dicts returned from retrieval
    role = 'doctor' | 'nurse' | 'patient' | 'admin'
    """

    filtered = []

    for r in retrieved_rows:
        if role == "doctor":
            filtered.append(r)

        elif role == "nurse":
            filtered.append({
                "filename": r.get("filename"),
                "impression": r.get("impression"),  
                "modality": r.get("modality"),
                "patient_id": r.get("patient_id"),
            })

        elif role == "patient":
            filtered.append({
                "filename": r.get("filename"),
                "explanation": simplify_text(r.get("impression","")),
            })

        elif role == "admin":
            filtered.append({
                "filename": r.get("filename"),
                "metadata": f"{r.get('modality')} | {r.get('filename')}"
            })

    return filtered

def simplify_text(text):
    """
    Basic linguistic simplification for patient-friendly explanation.
    """
    if text is None:
        return ""
    text = text.replace("cardiomegaly", "enlarged heart")
    text = text.replace("opacity", "shadow in the lung")
    text = text.replace("effusion", "fluid build-up")
    return text
