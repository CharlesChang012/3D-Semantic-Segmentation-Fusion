import torch
import torch.nn as nn
import torch.nn.functional as F
from LovaszSoftmax.pytorch.lovasz_losses import lovasz_softmax_flat
from typing import List, Optional

class CELSLoss(nn.Module):
    """
    Combines Cross-Entropy loss and Lovasz-Softmax loss for point-level segmentation.
    Handles masks for padded or missing points robustly.
    weight: class weights for Cross-Entropy 
    """
    def __init__(self, weight=None, ignore_index=0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.ignore_index = ignore_index    # noise class

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
        # print("pred_scores.shape:", pred_scores.shape)
        # print("gt_labels.shape:", gt_labels.shape)
        # print("mask shape:", mask.shape if mask is not None else "None")
        if mask is not None:
            # Ensure mask length matches predictions
            if mask.shape[1] > P:
                # Mask is longer (likely padded)
                print(f"[CELSLoss] Warning: mask longer ({mask.shape[1]}) than pred_scores ({P}), truncating mask.")
                mask = mask[:, :P]
            elif mask.shape[1] < P:
                # Predictions are longer (e.g., fewer valid points)
                print(f"[CELSLoss] Warning: pred_scores longer ({P}) than mask ({mask.shape[1]}), truncating preds.")
                pred_scores = pred_scores[:, :mask.shape[1]]
                gt_labels = gt_labels[:, :mask.shape[1]]
                P = mask.shape[1]

            # Flatten valid points
            mask_flat = mask.reshape(-1).to(torch.bool)
            pred_scores_flat = pred_scores.reshape(-1, C)[mask_flat]
            gt_labels_flat = gt_labels.reshape(-1)[mask_flat]
        else:
            pred_scores_flat = pred_scores.reshape(-1, C)
            gt_labels_flat = gt_labels.reshape(-1)

        # Further filter out ignored labels (noise label 0)
        valid_mask = gt_labels_flat != self.ignore_index
        pred_scores_flat_valid = pred_scores_flat[valid_mask]
        pred_probs_flat_valid = F.softmax(pred_scores_flat_valid, dim=-1)
        gt_labels_flat_valid = gt_labels_flat[valid_mask]   # [1-16]

        # Cross-Entropy Loss
        ce_loss = self.ce_loss(pred_scores_flat_valid, gt_labels_flat_valid - 1)  # shift labels to [0, C-1] for CE
        print("\nce_loss:", ce_loss.item(), "\n")

        # Lovasz Loss on full batch (B, P, C)
        lovasz_loss = lovasz_softmax_flat(pred_probs_flat_valid, gt_labels_flat_valid - 1)  # Lovasz expects labels in [0, C-1]
        print("lovasz_loss:", lovasz_loss.item(), "\n")

        # Calculate predictions
        predictions = torch.argmax(pred_probs_flat_valid, dim=-1) + 1  # shift back to original labels [1-16]

        total_loss = ce_loss + lovasz_loss

        return total_loss, ce_loss, lovasz_loss, predictions, gt_labels_flat_valid