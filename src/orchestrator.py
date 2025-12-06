# src/orchestrator.py
from tools.retrieval import retrieve_text, retrieve_image
from rbac import apply_rbac

def orchestrate(query, role="doctor", top_k_text=5, top_k_img=5):
    """
    1) Use retrieval tools
    2) Apply RBAC filtering
    3) Return structured dict for reasoner
    """
    text_hits = retrieve_text(query, k=top_k_text)
    img_hits = retrieve_image(query, k=top_k_img)

    retrieved = []

    # include text items first (findings + impression)
    for t in text_hits:
        retrieved.append({
            "filename": t["filename"],
            "patient_id": t["patient_id"],
            "modality": t["modality"],
            "findings": t.get("findings",""),
            "impression": t.get("impression",""),
            "score": t["score"],
            "source": "text"
        })

    # include image items
    for it in img_hits:
        retrieved.append({
            "filename": it["filename"],
            "patient_id": it["patient_id"],
            "modality": it["modality"],
            "findings": it.get("findings",""),
            "impression": it.get("impression",""),
            "score": it["score"],
            "source": "image"
        })

    # apply RBAC
    filtered = apply_rbac(role, retrieved)

    return {"query": query, "role": role, "retrieved": filtered}

if __name__ == "__main__":
    print(orchestrate("enlarged heart", role="doctor"))
