# agents/pathology_detection_agent.py

"""
Pathology Detection Agent for Multimodal RAG System
Integrates trained DenseNet model into the clinical pipeline
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add pathology detection to path
sys.path.append(str(Path(__file__).parent.parent))

from pathology_detection.models.densenet_classifier import build_model
from pathology_detection.training.config import cfg
from pathology_detection.training.dataset import get_val_transforms
from utils.logger import get_logger

logger = get_logger("PathologyDetector")


class PathologyDetectionAgent:
    """
    Production-ready pathology detection agent
    
    Features:
    - Loads trained model checkpoint
    - Provides probability scores for 14 pathologies
    - Returns top-k predictions above threshold
    - Handles errors gracefully
    """
    
    def __init__(self, checkpoint_path=None, threshold=0.40, top_k=5):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            threshold: Minimum probability to consider a pathology
            top_k: Return top K pathologies
        """
        
        self.threshold = threshold
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"[PathologyDetector] Initializing on device: {self.device}")
        
        # Load model
        if checkpoint_path is None:
            checkpoint_path = cfg.WEIGHTS_DIR / "best_model.pth"
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"[PathologyDetector] Checkpoint not found: {checkpoint_path}")
            logger.warning("[PathologyDetector] Detector will run in MOCK mode (returning zeros)")
            self.model = None
            self.mock_mode = True
        else:
            self.model = self._load_model(checkpoint_path)
            self.model.eval()
            self.mock_mode = False
            logger.info("[PathologyDetector] Model loaded successfully")
        
        # Transform
        self.transform = get_val_transforms(cfg.IMAGE_SIZE)
        
        # Pathology classes
        self.pathology_classes = cfg.PATHOLOGY_CLASSES
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        
        model = build_model(
            model_name=cfg.MODEL_NAME,
            num_classes=cfg.NUM_CLASSES
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"[PathologyDetector] Loaded checkpoint from epoch {checkpoint['epoch']}")
        logger.info(f"[PathologyDetector] Validation AUROC: {checkpoint['metrics']['auroc']:.4f}")
        
        return model
    
    def detect_pathologies(self, image_path: str) -> dict:
        """
        Detect pathologies in a chest X-ray image
        
        Args:
            image_path: Path to chest X-ray image
            
        Returns:
            Dictionary with pathology names and probability scores
            {
                'Atelectasis': 0.12,
                'Cardiomegaly': 0.03,
                'Effusion': 0.87,
                ...
            }
        """
        
        # MOCK MODE (if model not loaded)
        if self.mock_mode:
            return {pathology: 0.0 for pathology in self.pathology_classes}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Create result dictionary
            results = {
                pathology: float(prob)
                for pathology, prob in zip(self.pathology_classes, probabilities)
            }
            
            logger.debug(f"[PathologyDetector] Detected pathologies for: {Path(image_path).name}")
            
            return results
            
        except Exception as e:
            logger.error(f"[PathologyDetector] Error processing {image_path}: {e}")
            # Return zeros on error
            return {pathology: 0.0 for pathology in self.pathology_classes}
    
    def get_top_k_pathologies(
        self, 
        scores: dict, 
        k: int = None, 
        threshold: float = None
    ) -> list:
        """
        Get top-k detected pathologies above threshold
        
        Args:
            scores: Dictionary of pathology scores
            k: Number of top pathologies to return (default: self.top_k)
            threshold: Minimum score threshold (default: self.threshold)
            
        Returns:
            List of tuples [(pathology_name, probability), ...]
            Sorted by probability in descending order
        """
        
        if k is None:
            k = self.top_k
        if threshold is None:
            threshold = self.threshold
        
        # Filter by threshold
        filtered = {
            pathology: score 
            for pathology, score in scores.items() 
            if score >= threshold
        }
        
        # Sort by score
        sorted_items = sorted(
            filtered.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top-k
        return sorted_items[:k]
    
    def format_findings(self, scores: dict, top_k: int = None) -> str:
        """
        Format pathology findings as a clinical string
        
        Args:
            scores: Dictionary of pathology scores
            top_k: Number of findings to include
            
        Returns:
            Formatted string for clinical reasoning
        """
        
        top_pathologies = self.get_top_k_pathologies(scores, k=top_k)
        
        if not top_pathologies:
            return "No significant pathologies detected above threshold."
        
        findings = []
        for pathology, score in top_pathologies:
            confidence_pct = score * 100
            findings.append(f"- {pathology}: {confidence_pct:.1f}% confidence")
        
        return "\n".join(findings)
    
    def analyze_evidence(self, evidence: list) -> list:
        """
        Analyze all images in evidence list
        
        Args:
            evidence: List of evidence dictionaries (from retrieval)
            
        Returns:
            Updated evidence list with pathology scores added
        """
        
        logger.info(f"[PathologyDetector] Analyzing {len(evidence)} evidence items")
        
        for idx, e in enumerate(evidence):
            image_path = e.get("image_path")
            
            if image_path and Path(image_path).exists():
                # Detect pathologies
                scores = self.detect_pathologies(image_path)
                top_pathologies = self.get_top_k_pathologies(scores)
                
                # Add to evidence
                e["pathology_scores"] = scores
                e["top_pathologies"] = top_pathologies
                e["pathology_findings"] = self.format_findings(scores)
                
                logger.debug(f"[PathologyDetector] Evidence {idx+1}: {len(top_pathologies)} pathologies detected")
            else:
                # No image - set empty results
                e["pathology_scores"] = {}
                e["top_pathologies"] = []
                e["pathology_findings"] = "No image available for pathology detection."
        
        logger.info("[PathologyDetector] Analysis complete")
        return evidence


# Global singleton instance (lazy loading)
_pathology_detector = None

def get_pathology_detector():
    """Get or create global pathology detector instance"""
    global _pathology_detector
    
    if _pathology_detector is None:
        _pathology_detector = PathologyDetectionAgent()
    
    return _pathology_detector


if __name__ == "__main__":
    # Test the agent
    detector = PathologyDetectionAgent()
    
    # Test with a sample image
    test_image = "data/raw/dataset_indiana/images/images_normalized/1_IM-0001-4001.dcm.png"
    
    if Path(test_image).exists():
        scores = detector.detect_pathologies(test_image)
        print("\n" + "="*60)
        print("PATHOLOGY DETECTION TEST")
        print("="*60)
        print(f"\nImage: {test_image}")
        print("\nAll Scores:")
        for pathology, score in scores.items():
            print(f"  {pathology:20s}: {score:.4f}")
        
        print("\nTop Pathologies:")
        top = detector.get_top_k_pathologies(scores)
        for pathology, score in top:
            print(f"  {pathology:20s}: {score*100:.1f}%")
        
        print("\nFormatted Findings:")
        print(detector.format_findings(scores))
        print("="*60)
    else:
        print(f"Test image not found: {test_image}")