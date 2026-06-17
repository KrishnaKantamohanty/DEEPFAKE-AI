import torch
import torch.nn as nn
import torchvision.models as models

class FakeImageClassifier(nn.Module):
    """
    Deep Learning Model for Fake Image Identification.
    Utilizes a ResNet-18 backbone pre-trained on ImageNet, adapted for binary classification.
    
    Why ResNet-18 is effective for Fake/Deepfake Detection:
    1. Pre-trained Texture Detectors: ImageNet training equips the convolutional layers
       with sensitive filters for edges, shapes, and micro-textures. AI-generated images
       often contain structural inconsistencies, subtle blurring, or unusual noise patterns.
    2. Residual Connections: The skip-connections in ResNet allow fine-grained, high-frequency
       spatial gradient details to flow directly into deeper layers without vanishing. This is
       crucial for detecting sub-pixel GAN grid-like artifacts and checkerboard patterns.
    3. Computational Efficiency: ResNet-18 is relatively lightweight, allowing it to be trained
       efficiently on CPU systems while still retaining high performance.
    """
    def __init__(self, fine_tune=True, pretrained=True):
        """
        Args:
            fine_tune (bool): If True, all weights are trainable (recommended for deepfake tasks).
                              If False, only the final linear classification head is trained.
            pretrained (bool): If True, initialize the backbone with ImageNet weights.
        """
        super(FakeImageClassifier, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        if pretrained:
            # Prefer ImageNet weights, but allow the project to run in offline or
            # permission-restricted environments where Torch cannot download them.
            try:
                self.backbone = models.resnet18(weights=weights)
            except Exception as exc:
                print(f"Warning: could not load pretrained ResNet-18 weights ({exc}).")
                print("Continuing with randomly initialized ResNet-18 weights.")
                self.backbone = models.resnet18(weights=None)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Extract input size of original fully connected head
        in_features = self.backbone.fc.in_features
        
        # Replace the fully connected layer with a customized head
        # We output 1 raw logit, which is highly stable when combined with BCEWithLogitsLoss.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),              # Regularization to prevent overfitting
            nn.Linear(in_features, 1)       # Binary logit output (<=0: REAL, >0: FAKE)
        )
        
        # Configure weight freezing
        if not fine_tune:
            # Freeze all parameters in the backbone
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        else:
            # Fine-tune the entire model (standard practice for subtle fake features)
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Runs the forward pass.
        Returns raw logits (pre-sigmoid).
        """
        return self.backbone(x)


def get_model(fine_tune=True, device='cpu', pretrained=True):
    """
    Instantiates the model and moves it to the appropriate device.
    """
    model = FakeImageClassifier(fine_tune=fine_tune, pretrained=pretrained)
    model = model.to(device)
    return model
