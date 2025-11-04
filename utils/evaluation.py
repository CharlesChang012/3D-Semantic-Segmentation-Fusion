import torch
import time
import numpy as np

def compute_confusion_matrix(preds, labels, num_classes):
    mask = (labels >= 0) & (labels < num_classes)
    hist = torch.bincount(
        num_classes * labels[mask] + preds[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


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