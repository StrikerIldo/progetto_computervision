import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: output della rete (logits), shape [Batch, 1, H, W]
        # targets: ground truth, shape [Batch, 1, H, W]
        
        # Applichiamo sigmoide per portare i logits tra 0 e 1
        inputs = torch.sigmoid(inputs)
        
        # Appiattiamo i tensori (flatten) per calcolare l'intersezione su tutti i pixel
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Formula Dice: 2 * (Intersection) / (Union)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        # La Loss è 1 - Dice Score (perché vogliamo minimizzarla)
        return 1 - dice

def combined_loss(pred, target, bce_weight=0.5):
    """
    Combinazione di Binary Cross Entropy e Dice Loss come suggerito dalle specifiche.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = DiceLoss()(pred, target)
    
    return bce * bce_weight + dice * (1 - bce_weight)