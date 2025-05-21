from sklearn.metrics import roc_auc_score, roc_curve
import torch
import numpy as np


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum("bgf,cf->bcg", [features, tweight])
    return cam_maps


# def roc_threshold(label, prediction):
#     fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
#     fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
#     c_auc = roc_auc_score(label, prediction)
#     return c_auc, threshold_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


# def eval_metric(oprob, label):

#     auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
#     prob = oprob > threshold
#     label = label > threshold

#     TP = (prob & label).sum(0).float()
#     TN = ((~prob) & (~label)).sum(0).float()
#     FP = (prob & (~label)).sum(0).float()
#     FN = ((~prob) & label).sum(0).float()

#     accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
#     precision = torch.mean(TP / (TP + FP + 1e-12))
#     recall = torch.mean(TP / (TP + FN + 1e-12))
#     specificity = torch.mean( TN / (TN + FP + 1e-12))
#     F1 = 2*(precision * recall) / (precision + recall+1e-12)

#     return accuracy, precision, recall, specificity, F1, auc


def roc_threshold(label, prediction):
    # Check if there's only one class in the labels
    if len(np.unique(label)) == 1:
        print("Warning: Only one class present in labels. ROC AUC not defined.")
        return 0.0, 0.5  # Return default threshold

    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal


def eval_metric(oprob, label):
    label_np = label.cpu().numpy()
    oprob_np = oprob.detach().cpu().numpy()

    # Check for single class case
    if len(np.unique(label_np)) == 1:
        print("Warning: Only one class in evaluation set. Using default metrics.")
        single_class = np.unique(label_np)[0]

        # Return default metrics for single-class case
        accuracy = 1.0  # All predictions match the single class
        precision = 1.0 if single_class == 1 else 0.0
        recall = 1.0 if single_class == 1 else 0.0
        specificity = 1.0 if single_class == 0 else 0.0
        f1 = 1.0 if single_class == 1 else 0.0
        auc = 0.0  # AUC not defined for single class

        return accuracy, precision, recall, specificity, f1, auc

    # Normal case with multiple classes
    auc, threshold = roc_threshold(label_np, oprob_np)
    pred = (oprob_np > threshold).astype(int)

    # Calculate metrics
    tp = np.sum((pred == 1) & (label_np == 1))
    tn = np.sum((pred == 0) & (label_np == 0))
    fp = np.sum((pred == 1) & (label_np == 0))
    fn = np.sum((pred == 0) & (label_np == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return accuracy, precision, recall, specificity, f1, auc
