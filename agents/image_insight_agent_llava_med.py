# agents/image_insight_agent_llava_med.py

import requests
import base64
import re
from utils.logger import get_logger

logger = get_logger("LLaVA-Med")

OLLAMA_URL = "http://localhost:11434/api/generate"
# Ensure this matches your actual model name in Ollama
LLAVA_MED_MODEL = "z-uo/llava-med-v1.5-mistral-7b_q8_0" 

# -------------------------------
# Utility
# -------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------------------
# Main Agent
# -------------------------------

def image_insight_agent_llava_med(evidence: list, query: str) -> list:
    """
    Generates an insight using LLaVA-Med, GUIDED by the retrieved Pattern Recognition evidence.
    """
    image_insights = []

    # =========================================================================
    # [NOVELTY INTEGRATION]: Extract Pattern Recognition Evidence
    # =========================================================================
    
    pattern_evidence_text = ""
    if evidence:
        pattern_evidence_text = "\n[PATTERN RECOGNITION ANALYSIS]\n"
        pattern_evidence_text += "The system analyzed the visual patterns (opacities, texture, size) and found these similar historical cases:\n"
        
        # We look at the top 3 retrieved matches to guide the LLM
        for i, ev in enumerate(evidence[:3]): 
            # --- FIX: Handle Dictionary Access Safely ---
            diagnosis = "Unknown Condition"
            
            # Check if 'ev' is a dictionary (Standard LangGraph behavior)
            if isinstance(ev, dict):
                # Try getting 'MeSH' from 'payload' dict
                payload = ev.get('payload', {})
                if payload:
                    diagnosis = payload.get('MeSH', 'Unknown Condition')
                else:
                    # If payload is flattened or missing
                    diagnosis = ev.get('MeSH', 'Unknown Condition')
            else:
                # Fallback if 'ev' is still an Object (Unlikely but safe)
                if hasattr(ev, 'payload'):
                    diagnosis = getattr(ev.payload, 'get', lambda k,d: d)('MeSH', 'Unknown Condition')
            
            pattern_evidence_text += f"- Match {i+1}: Historically diagnosed with '{diagnosis}' (High Visual Correlation)\n"

    # =========================================================================

    for idx, e in enumerate(evidence, start=1):
        # Handle dict access for image path as well
        image_path = e.get("image_path") if isinstance(e, dict) else getattr(e, "image_path", None)

        if not image_path:
            logger.debug(f"[LLaVA-Med] Skipping evidence {idx}: no image")
            continue

        logger.info(f"[LLaVA-Med] Analyzing image {idx}: {image_path}")

        try:
            image_b64 = encode_image(image_path)
        except Exception as ex:
            logger.error(f"[LLaVA-Med] Image load failed: {ex}")
            continue

        # -------------------------------
        # NOVELTY PROMPT (Injecting the Evidence)
        # -------------------------------
        prompt = f"""
Act as a Senior Radiologist. 
CLINICAL QUERY: "{query}"
Also provided with the pattern recognition analysis output:{pattern_evidence_text}
Analyze the attached X-Ray image.
Provide a purely observational report (Anatomy & Findings). 
"""

        payload = {
            "model": LLAVA_MED_MODEL,
            "prompt": prompt.strip(),
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1 
            }
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            insight = response.json()["response"]
            image_insights.append(f"[R{idx}-IMAGE] {insight}")
        except Exception as e:
            logger.error(f"LLaVA request failed: {e}")
            image_insights.append(f"[Error] Could not generate insight.")

    return image_insights