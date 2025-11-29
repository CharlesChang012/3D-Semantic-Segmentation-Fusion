import torch
import time
import numpy as np

def evaluate(all_preds, all_labels, num_classes, total_loss, total_correct, total_points, iteration):

    # Evaluate metrics
    conf_mat = compute_confusion_matrix(all_preds, all_labels, num_classes)
    iou_per_class, miou = compute_iou(conf_mat)
    acc_per_class, mean_acc = per_class_accuracy(conf_mat)
    precision, recall, f1 = precision_recall_f1(conf_mat)

    print("\n====== EVALUATION METRICS ======")
    print(f"Loss: {total_loss/iteration:.4f}, Overall Acc: {total_correct/total_points:.4f}")

    print(f"Per-Class Acc: {[f'{v:.4f}' for v in acc_per_class.tolist()]}, Mean Per-Class Acc: {mean_acc.item():.4f}")
    print(f"Per-Class IoU: {[f'{v:.4f}' for v in iou_per_class.tolist()]}, Mean IoU: {miou.item():.4f}")

    print(f"Precision: {precision.item():.4f}, Recall: {recall.item():.4f}, F1: {f1.item():.4f}")
    print("=================================\n")


    return {
        'loss': total_loss / iteration,
        'overall_acc': total_correct / total_points,
        'iou_per_class': iou_per_class.tolist(),
        'mean_iou': miou.item(),
        'mean_per_class_acc': mean_acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def compute_confusion_matrix(preds, labels, num_classes):
    preds = preds - 1
    labels = labels - 1
    conf_matrix = torch.bincount(
        num_classes * labels + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return conf_matrix


def compute_iou(conf_matrix):
    intersection = torch.diag(conf_matrix)
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = intersection.float() / torch.clamp(union.float(), min=1)
    mean_iou = torch.mean(iou)
    return iou, mean_iou


def per_class_accuracy(conf_matrix):
    acc = torch.diag(conf_matrix).float() / torch.clamp(conf_matrix.sum(1).float(), min=1)
    mean_acc = torch.mean(acc)
    return acc, mean_acc


def overall_accuracy(conf_matrix):
    correct = torch.diag(conf_matrix).sum().float()
    total = conf_matrix.sum().float()
    return correct / torch.clamp(total, min=1)


def precision_recall_f1(conf_matrix):
    tp = torch.diag(conf_matrix).float()
    fp = conf_matrix.sum(0).float() - tp
    fn = conf_matrix.sum(1).float() - tp

    precision = torch.mean(tp / torch.clamp(tp + fp, min=1))
    recall = torch.mean(tp / torch.clamp(tp + fn, min=1))
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-6)
    return precision, recall, f1


def measure_efficiency(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    runtime = time.time() - start_time
    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    return result, {'runtime_sec': runtime, 'mem_alloc_MB': mem_alloc}
