# evaluation/diagnosis_evaluator.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# =========================================================
# STANDALONE FUNCTIONS (For Agents & Backward Compatibility)
# =========================================================

# Global instance to share memory
_shared_embedder = None

def _get_embedder():
    global _shared_embedder
    if _shared_embedder is None:
        print("Loading Shared Evaluator Model...")
        _shared_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _shared_embedder

def precision_recall_mrr(retrieved, ground_truth, k=7):
    # Ensure uniqueness
    relevant = set(ground_truth)
    retrieved_k = retrieved[:k]
    retrieved_unique = set(retrieved_k)

    # True positives = unique relevant items retrieved
    tp = len(retrieved_unique & relevant)

    # Use actual retrieved count or k, whichever is smaller
    actual_k = min(k, len(retrieved_k))

    precision = tp / actual_k if actual_k > 0 else 0
    recall = tp / len(relevant) if relevant else 0

    # MRR: first relevant hit
    mrr = 0
    for idx, r in enumerate(retrieved_k, start=1):
        if r in relevant:
            mrr = 1 / idx
            break

    return {
        "Precision@K": round(precision, 3),
        "Recall@K": round(recall, 3),
        "MRR": round(mrr, 3)
    }

def groundedness(answer):
    lines = [
        l.strip() for l in answer.split("\n")
        if l.strip()
        and not l.lower().startswith("supporting evidence")
        and not l.lower().startswith("next steps")
        and not l.lower().startswith("diagnosis")
    ]

    factual_lines = []
    for l in lines:
        if l.startswith("-") or ":" in l:
            factual_lines.append(l)

    if not factual_lines:
        return 0

    supported = sum(
        1 for l in factual_lines
        if re.search(r"\[(R\d+|Rx)(-IMAGE)?\]", l)
    )

    return round(supported / len(factual_lines), 3)

def clinical_correctness(answer, impressions, threshold=0.55):
    if not impressions:
        return 0.0

    embedder = _get_embedder()
    
    # Ensure inputs are strings
    answer_str = str(answer)
    imp_list = [str(i) for i in impressions] if isinstance(impressions, list) else [str(impressions)]

    answer_emb = embedder.encode([answer_str])
    imp_embs = embedder.encode(imp_list)

    sims = cosine_similarity(answer_emb, imp_embs)[0]
    return float(np.max(sims))

def completeness(answer):
    score = 0
    if "[R" in answer:
        score += 1
    if any(w in answer.lower() for w in [
        "recommend", "next step", "follow-up",
        "no further action required", "no further action"
    ]):
        score += 1
    if "diagnosis" in answer.lower():
        score += 1
    return round(score / 3, 2)

# =========================================================
# CLASS-BASED EVALUATOR (For Comparison Script)
# =========================================================

class DiagnosisEvaluator:
    def __init__(self):
        # Reuse shared embedder
        self.embedder = _get_embedder()

    def evaluate(self, answer, ground_truth):
        """
        Main entry point for the comparison script.
        Returns dictionary of metrics.
        """
        # 1. Semantic Similarity (BERT Score equivalent)
        bert_sim = clinical_correctness(answer, ground_truth)
        
        # 2. Approximate Text Overlap Metrics
        bleu = self._simple_unigram_precision(answer, ground_truth)
        rouge = self._simple_unigram_recall(answer, ground_truth)

        return {
            "bert_similarity": round(bert_sim, 4),
            "bleu_score": round(bleu, 4),
            "rouge1": round(rouge, 4),
            "clinical_accuracy": round(bert_sim, 4) # Using similarity as proxy
        }

    def _simple_unigram_precision(self, candidate, reference):
        """Approximation of BLEU-1 (Precision)"""
        c_tokens = str(candidate).lower().split()
        r_tokens = str(reference).lower().split()
        
        if not c_tokens: return 0.0
        
        matches = sum(1 for w in c_tokens if w in r_tokens)
        return matches / len(c_tokens)

    def _simple_unigram_recall(self, candidate, reference):
        """Approximation of ROUGE-1 (Recall)"""
        c_tokens = str(candidate).lower().split()
        r_tokens = str(reference).lower().split()
        
        if not r_tokens: return 0.0
        
        matches = sum(1 for w in c_tokens if w in r_tokens)
        return matches / len(r_tokens)