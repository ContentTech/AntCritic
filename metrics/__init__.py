import sys
import numpy as np
import sklearn.metrics as metrics
import torch


def calc_grid_metric(pred, label, **kwargs):
    pred, label = pred.cpu().numpy(), label.cpu().numpy()
    macro_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="macro", labels=[1, 2, 3])
    all_macro = metrics.f1_score(y_pred=pred, y_true=label, average="macro")
    micro_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="micro", labels=[1, 2, 3])
    all_micro = metrics.f1_score(y_pred=pred, y_true=label, average="micro")
    weighted_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="weighted", labels=[1, 2, 3])
    all_weighted = metrics.f1_score(y_pred=pred, y_true=label, average="weighted")
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "all_macro": all_macro,
        "all_micro": all_micro,
        "all_weighted": all_weighted
    }

def calc_major_metric(is_major, label, confidence):
    # in (ALL, L)
    masked_confidence = confidence.masked_fill(label != 1, -1e9).softmax(-1)
    major_confidence = masked_confidence.masked_fill(~is_major, 0).sum(-1)  # (ALL)
    return {
        "mean": major_confidence.mean(0)
    }


def calc_sentence_metric(pred, label, **kwargs):
    pred, label = pred.cpu().numpy(), label.cpu().numpy()
    macro_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="macro", labels=[1, 2])
    all_macro = metrics.f1_score(y_pred=pred, y_true=label, average="macro")
    micro_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="micro", labels=[1, 2])
    all_micro = metrics.f1_score(y_pred=pred, y_true=label, average="micro")
    weighted_f1 = metrics.f1_score(y_pred=pred, y_true=label, average="weighted", labels=[1, 2])
    all_weighted = metrics.f1_score(y_pred=pred, y_true=label, average="weighted")
    # valid_mask = target != 0
    # valid_f1 = metrics.f1_score(y_pred=(pred != 0), y_true=(target != 0))
    # claim_f1 = metrics.f1_score(y_pred=(pred != 1), y_true=(target != 1))
    # claim_valid_f1 = metrics.f1_score(y_pred=(pred[valid_mask] == 1), y_true=(target[valid_mask] == 1))
    # premise_valid_f1 = metrics.f1_score(y_pred=(pred[valid_mask] == 2), y_true=(target[valid_mask] == 2))
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "all_macro": all_macro,
        "all_micro": all_micro,
        "all_weighted": all_weighted
        # "valid_f1": valid_f1,
        # "claim_f1": claim_valid_f1,
        # "premise_f1": premise_valid_f1
    }

def calc_first_metric(*input, **kwargs):
    return {
        "item": calc_sentence_metric(*input, **kwargs)
    }


def calc_second_metric(pred_label, real_label, pred_grid, real_grid, is_major, full_label, major_logits):
    return {
        "label": calc_sentence_metric(pred_label, real_label),
        "grid": calc_grid_metric(pred_grid, real_grid),
        "major": calc_major_metric(is_major, full_label, major_logits)
    }