# pathology_detection/models/densenet_classifier.py

"""
DenseNet-121 classifier for multi-label pathology detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List


class DenseNetClassifier(nn.Module):
    """
    DenseNet-121 based multi-label classifier
    
    Architecture:
        - DenseNet-121 backbone (pretrained on ImageNet)
        - Global Average Pooling
        - Fully Connected layer (1024 -> num_classes)
        - Sigmoid activation for multi-label output
    """
    
    def __init__(
        self, 
        num_classes: int = 14, 
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(DenseNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load DenseNet-121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Freeze backbone if specified (faster training, less overfitting)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from backbone
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes)
        )
        
        # Sigmoid for multi-label classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            logits: Output tensor [batch_size, num_classes]
        """
        logits = self.backbone(x)
        return logits
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities instead of logits
        
        Returns:
            probs: Probabilities [batch_size, num_classes]
        """
        logits = self.forward(x)
        probs = self.sigmoid(logits)
        return probs
    
    def predict_with_threshold(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get binary predictions with threshold
        
        Returns:
            predictions: Binary tensor [batch_size, num_classes]
        """
        probs = self.predict_probabilities(x)
        predictions = (probs >= threshold).float()
        return predictions


class ResNetClassifier(nn.Module):
    """
    Alternative: ResNet-50 classifier
    """
    
    def __init__(
        self, 
        num_classes: int = 14, 
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super(ResNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = models.resnet50(pretrained=pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return logits
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = self.sigmoid(logits)
        return probs


def build_model(
    model_name: str = "densenet121",
    num_classes: int = 14,
    pretrained: bool = True,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Factory function to build model
    
    Args:
        model_name: "densenet121" or "resnet50"
        num_classes: Number of pathology classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone for transfer learning
        
    Returns:
        model: PyTorch model
    """
    
    if model_name == "densenet121":
        model = DenseNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_name == "resnet50":
        model = ResNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = build_model("densenet121", num_classes=14)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    probs = model.predict_probabilities(dummy_input)
    
    print("Model Architecture:")
    print(model)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"\nSample probabilities:\n{probs[0]}")