# src/evaluation.py
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

def safe_text(t):
    return t.strip() if isinstance(t, str) else ""

def evaluate_all(reference, prediction, retrieved):
    print("\n================ EVALUATION DEBUG ================")
    print("REFERENCE TEXT:", reference)
    print("PREDICTION TEXT:", prediction)

    if not reference:
        print("⚠ No reference → BLEU / ROUGE skipped.")
        ref_tokens = []
    else:
        ref_tokens = reference.split()

    # BLEU
    bleu = 0.0
    try:
        bleu = sentence_bleu([ref_tokens], prediction.split()) if ref_tokens else 0.0
    except:
        bleu = 0.0

    print(f"BLEU: {bleu:.4f}")

    # ROUGE-L
    rouge_l = 0.0
    if reference:
        try:
            r = Rouge()
            rouge_l = r.get_scores(prediction, reference)[0]["rouge-l"]["f"]
        except:
            rouge_l = 0.0
    print(f"ROUGE-L: {rouge_l:.4f}")

    # Recall@K
    gold = safe_text(reference).lower()
    rec1 = any(gold in (item.get("findings","")+item.get("impression","")).lower() for item in retrieved)
    print(f"Recall@1: {1.0 if rec1 else 0.0}")

    print("=====================================================\n")
