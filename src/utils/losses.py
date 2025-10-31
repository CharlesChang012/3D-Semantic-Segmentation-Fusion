import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses import lovasz_softmax

class CELSLoss(nn.Module):
    """
    Combines Cross-Entropy loss and Lovasz-Softmax loss for point-level segmentation.
    Handles masks for padded points.
    """
    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, pred_scores, gt_labels, mask=None):
        """
        Args:
            pred_scores: (B, P, C) - logits
            gt_labels: (B, P) - ground truth
            mask: (B, P) optional - True for valid points
        Returns:
            total_loss, ce_loss, lovasz_loss
        """
        B, P, C = pred_scores.shape

        # Flatten valid points for Cross-Entropy
        if mask is not None:
            mask_flat = mask.view(-1)
            pred_scores_flat = pred_scores.view(-1, C)[mask_flat]   # (N_valid, C)
            gt_labels_flat = gt_labels.view(-1)[mask_flat]          # (N_valid,)
        else:
            pred_scores_flat = pred_scores.view(-1, C)
            gt_labels_flat = gt_labels.view(-1)


        # Cross-Entropy Loss
        ce_loss = self.ce_loss(pred_scores_flat, gt_labels_flat)

        # Lovasz loss works on (B, P, C)
        probs = F.softmax(pred_scores, dim=-1)
        lovasz_loss = lovasz_softmax(probs, gt_labels, ignore=self.ignore_index)

        total_loss = ce_loss + lovasz_loss
        return total_loss, ce_loss, lovasz_loss, preds_masked
