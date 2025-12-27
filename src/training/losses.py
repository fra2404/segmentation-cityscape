import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        
        # Create one-hot encoding manually for compatibility
        num_classes = probas.shape[1]
        targets_one_hot = torch.zeros_like(probas)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        
        intersection = torch.sum(probas * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probas + targets_one_hot, dim=(2, 3))
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice_score).mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=0.5, class_weights=None, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
