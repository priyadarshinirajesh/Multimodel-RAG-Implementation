# agents/verifiers/evidence_quality_verifier.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.logger import get_logger

logger = get_logger("EvidenceQualityVerifier")

_embedder = SentenceTransformer("all-MiniLM-L6-v2")


class EvidenceQualityVerifier:
    """Pre-filters evidence before sending to reasoning agent."""
    
    def __init__(self):
        self.relevance_threshold = 0.3
        self.min_evidence_count = 2
        self.max_evidence_count = 7
    
    def verify_and_filter(self, query: str, evidence: list, selected_modalities: list) -> dict:
        """Main verification and filtering function."""
        
        logger.info(f"[EvidenceFilter] Verifying {len(evidence)} evidence items...")
        
        if not evidence:
            return {
                "filtered_evidence": [],
                "quality_score": 0.0,
                "removed_count": 0,
                "needs_retrieval_adjustment": True,
                "feedback": "No evidence retrieved - increase retrieval limit"
            }
        
        # Step 1: Remove modality mismatches
        modality_filtered = self._filter_by_modality(evidence, selected_modalities)
        
        # Step 2: Calculate relevance scores
        scored_evidence = self._score_relevance(query, modality_filtered)
        
        # Step 3: Filter by relevance threshold
        relevant_evidence = [
            e for e in scored_evidence
            if e.get("relevance_score", 0) >= self.relevance_threshold
        ]
        
        # Step 4: Remove duplicates
        deduplicated = self._remove_duplicates(relevant_evidence)
        
        # Step 5: Rank and limit to top K
        ranked = sorted(deduplicated, key=lambda x: x.get("relevance_score", 0), reverse=True)
        final_evidence = ranked[:self.max_evidence_count]
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(final_evidence, len(evidence))
        removed_count = len(evidence) - len(final_evidence)
        
        # Determine if retrieval needs adjustment
        needs_adjustment = len(final_evidence) < self.min_evidence_count
        
        feedback = self._generate_feedback(
            original_count=len(evidence),
            final_count=len(final_evidence),
            quality_score=quality_score
        )
        
        logger.info(f"[EvidenceFilter] Filtered: {len(evidence)} → {len(final_evidence)} (removed {removed_count})")
        logger.info(f"[EvidenceFilter] Quality score: {quality_score:.2f}")
        
        return {
            "filtered_evidence": final_evidence,
            "quality_score": quality_score,
            "removed_count": removed_count,
            "needs_retrieval_adjustment": needs_adjustment,
            "feedback": feedback
        }
    
    def _filter_by_modality(self, evidence: list, allowed_modalities: list) -> list:
        """Hard filter: remove evidence from wrong modalities"""
        filtered = [
            e for e in evidence
            if e.get("modality") in allowed_modalities
        ]
        
        removed = len(evidence) - len(filtered)
        if removed > 0:
            logger.info(f"[EvidenceFilter] Removed {removed} items due to modality mismatch")
        
        return filtered
    
    def _score_relevance(self, query: str, evidence: list) -> list:
        """Calculate relevance scores using embeddings"""
        
        if not evidence:
            return []
        
        # Embed query
        query_embedding = _embedder.encode([query])[0]
        
        # Embed evidence texts
        evidence_texts = [e.get("report_text", "") for e in evidence]
        evidence_embeddings = _embedder.encode(evidence_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], evidence_embeddings)[0]
        
        # Add scores to evidence
        for e, score in zip(evidence, similarities):
            e["relevance_score"] = float(score)
        
        return evidence
    
    def _remove_duplicates(self, evidence: list) -> list:
        """
        Remove duplicate clinical statements across evidence,
        WITHOUT removing entire evidence objects.
        Images are untouched.
        """

        if len(evidence) <= 1:
            return evidence

        # Step 1: Extract all sentences with back-references
        sentence_bank = []
        for idx, e in enumerate(evidence):
            text = e.get("report_text", "")
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            for s in sentences:
                sentence_bank.append({
                    "sentence": s,
                    "evidence_idx": idx
                })

        if not sentence_bank:
            return evidence

        # Step 2: Embed all sentences
        sentences = [item["sentence"] for item in sentence_bank]
        embeddings = _embedder.encode(sentences)

        similarity_matrix = cosine_similarity(embeddings)

        # Step 3: Identify duplicate sentences
        duplicate_sentence_indices = set()
        for i in range(len(sentences)):
            for j in range(i):
                if similarity_matrix[i][j] > 0.92:  # strict threshold
                    duplicate_sentence_indices.add(i)
                    break

        # Step 4: Reconstruct cleaned reports
        cleaned_reports = {i: [] for i in range(len(evidence))}

        for idx, item in enumerate(sentence_bank):
            if idx not in duplicate_sentence_indices:
                cleaned_reports[item["evidence_idx"]].append(item["sentence"])

        # Step 5: Update evidence objects
        for idx, e in enumerate(evidence):
            cleaned_text = ". ".join(cleaned_reports[idx])
            if cleaned_text:
                e["report_text"] = cleaned_text + "."
            else:
                e["report_text"] = ""

        return evidence

    
    def _calculate_quality_score(self, filtered_evidence: list, original_count: int) -> float:
        """Calculate overall evidence quality score"""
        
        if not filtered_evidence:
            return 0.0
        
        # Component 1: Average relevance score
        avg_relevance = np.mean([e.get("relevance_score", 0) for e in filtered_evidence])
        
        # Component 2: Evidence diversity (different modalities is good)
        modalities = set(e.get("modality") for e in filtered_evidence)
        diversity_score = min(len(modalities) / 3.0, 1.0)
        
        # Component 3: Coverage (enough evidence without too much noise)
        coverage_score = min(len(filtered_evidence) / 5.0, 1.0)
        
        # Component 4: Filtering efficiency (not removing too much)
        retention_rate = len(filtered_evidence) / original_count if original_count > 0 else 0
        efficiency_score = 1.0 if retention_rate > 0.5 else retention_rate * 2
        
        # Weighted average
        quality_score = (
            avg_relevance * 0.4 +
            diversity_score * 0.2 +
            coverage_score * 0.2 +
            efficiency_score * 0.2
        )
        
        return round(quality_score, 3)
    
    def _generate_feedback(self, original_count: int, final_count: int, quality_score: float) -> str:
        """Generate actionable feedback for retrieval adjustment"""
        
        if quality_score >= 0.7:
            return "Evidence quality is good"
        
        if final_count < self.min_evidence_count:
            return f"Insufficient evidence ({final_count}/{self.min_evidence_count}) - increase retrieval limit"
        
        removal_rate = (original_count - final_count) / original_count if original_count > 0 else 0
        
        if removal_rate > 0.7:
            return "Too much irrelevant evidence - refine retrieval query"
        
        if quality_score < 0.5:
            return "Low quality evidence - consider alternative modalities or query reformulation"
        
        return "Evidence quality is acceptable but could be improved"