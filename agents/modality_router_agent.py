# agents/modality_router_agent.py

import requests
from utils.logger import get_logger

logger = get_logger("RouterAgent")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:instruct"   

def route_modalities(query: str):
    logger.info(f"Received query: {query}")

    prompt = f"""
You are a medical query routing assistant.

You MUST choose imaging modalities
ONLY from the datasets that are AVAILABLE in this system.

AVAILABLE DATASETS (STRICT):
1. XRAY → Chest X-ray dataset  
   - Covers: lungs, chest, pleura, heart, cardiomegaly, effusion, pneumonia, pneumothorax

2. CT → Pancreas CT dataset  
   - Covers: pancreas ONLY
   - Includes: pancreatitis, pancreatic mass, duct dilation

3. MRI → Prostate MRI dataset  
   - Covers: prostate ONLY
   - Includes: PI-RADS lesions, prostate cancer screening, BPH

NO OTHER DATA EXISTS.
There is:
- NO lab data
- NO vitals
- NO medications
- NO imaging outside these three datasets

TASK:
Determine which dataset(s) can answer the user’s question.

ROUTING RULES (VERY IMPORTANT):
- If the query clearly matches ONE dataset → return that modality only
- If the query is imaging-related BUT ambiguous across datasets → return ALL modalities
- If the query cannot be answered using imaging → return NO_IMAGING
- Do NOT guess organs that are not present in the datasets
- Do NOT include unnecessary modalities

CLINICAL MAPPING RULES:
- Chest, lungs, pleura, effusion, heart → XRAY
- Pancreas or pancreatic disease → CT
- Prostate or prostate cancer → MRI

AMBIGUITY RULE:
If you cannot confidently map the query to exactly one dataset,
but it still refers to medical imaging,
return ALL modalities:
["XRAY", "CT", "MRI"]

USER QUERY:
"{query}"

OUTPUT FORMAT (STRICT):
Return ONLY a valid JSON array.
No explanations.
No markdown.
No extra text.

VALID OUTPUTS:
["XRAY"]
["CT"]
["MRI"]
["XRAY", "CT", "MRI"]
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        raw_output = response.json()["response"].strip()

        # Safety parsing
        modalities = eval(raw_output)

        if not isinstance(modalities, list):
            raise ValueError("Invalid modality format")

        logger.info(f"Selected modalities (LLM): {modalities}")
        return modalities
    
    except Exception as e:
        logger.error(f"Router LLM failed, fallback to XRAY. Error: {e}")
        return ["XRAY"]
