# src/06_rag_answer.py

import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/flan-t5-base"    # lightweight but good
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def make_rag_answer(query, retrieved_context):
    """
    Combine user query + retrieved medical info and
    generate a summarized answer.
    """

    context_text = ""

    for item in retrieved_context:
        for k, v in item.items():
            context_text += f"{k}: {v}\n"
        context_text += "\n"

    prompt = f"""
You are a medical assistant. Use ONLY the information provided below.

Query: {query}

Retrieved Medical Records:
{context_text}

Generate a concise, safe, factual medical explanation.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=350)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

