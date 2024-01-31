import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from metrics import Stixel, EvaluationMetric
from typing import List


def calculate_stixel_iou(pred: Stixel, target: Stixel):
    overlap_start = max(target.top, pred.top)
    overlap_end = min(target.bottom, pred.bottom)
    intersection = max(0, overlap_end - overlap_start)
    total_height_pred = pred.bottom - pred.top
    total_height_target = target.bottom - target.top
    union = total_height_pred + total_height_target - intersection
    iou = intersection / union if union != 0 else 0
    return iou


class PrecisionRecall(EvaluationMetric):
    # 1. Implement __init()__
    def __init__(self, iou_threshold: float = 0.5, rm_used=False):
        super().__init__()
        self.iou_threshold: float = iou_threshold
        self.rm_used: bool = rm_used
        self.precision: float = 0.0
        self.recall: float = 0.0

    # 2. Implement evaluate()
    def evaluate(self, prediction, target):
        """
        Evaluate Stixels with multiple stixels per column using IoU, precision, and recall.
        """
        total_predicted: int = len(prediction)
        total_ground_truth: int = len(target)
        hits = self.find_best_matches(prediction, target)
        # hits equals True positives (TP)
        # precision = TP / TP + FP
        self.precision = hits / total_predicted if total_predicted != 0 else 0
        if self.precision > 1.0:
            self.precision = 1.0
        # len pred equals True positives + False positives (FP)
        # recall = TP / TP + FN
        self.recall = hits / total_ground_truth if total_ground_truth != 0 else 0
        # len gt equals True positives + False negatives (FN)
        return self.precision, self.recall

    # 2.1 Implement get_score()
    def get_score(self):
        if self.precision + self.recall == 0:
            return 0
        f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1_score

    def find_best_matches(self, predicted_stixels: List[Stixel], ground_truth_stixels: List[Stixel]):
        hits = 0
        remaining_pred_stixels = set(predicted_stixels)
        # iterate over all GT
        for gt_stixel in ground_truth_stixels:
            matched_pred = None
            # iterate over all predictions
            for pred_stixel in remaining_pred_stixels:
                # iterate over all columns
                if pred_stixel.column == gt_stixel.column:
                    # if iou score is high enough; count
                    iou_score = calculate_stixel_iou(pred_stixel, gt_stixel)
                    if iou_score >= self.iou_threshold:
                        matched_pred = pred_stixel
                        hits += 1
                        break
            if matched_pred:
                if self.rm_used:
                    remaining_pred_stixels.remove(matched_pred)
        return hits


def plot_precision_recall_curve(recall, precision):
    # Plotting the PR Curve
    f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision, recall)]
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, label='PR Curve')
    plt.plot(recall, f1_scores, label='F1-Score Curve', color='orange', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision / F1-Score')
    plt.title(f'Precision-Recall and F1-Score Curves')
    plt.xlim([0, 1])  # Set x-axis limits
    plt.ylim([0, 1])  # Set y-axis limits
    plt.legend()
    plt.show()


def draw_stixel_on_image_prcurve(image, best_matches, preds, stixel_width=8):
    image = np.array(image.numpy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for pred in preds:
        cv2.rectangle(image,
                      (pred.column, pred.top),
                      (pred.column + stixel_width, pred.bottom),
                      (255, 62, 150), 1)
    for match in best_matches.values():
        gt_stixel = match['target_stixel']
        pred_stixel = match['pred_stixel']

        overlap_top = max(gt_stixel.top, pred_stixel.top)
        overlap_bottom = min(gt_stixel.bottom, pred_stixel.bottom)
        overlap_left = max(gt_stixel.column, pred_stixel.column)
        overlap_right = min(gt_stixel.column + stixel_width, pred_stixel.column + stixel_width)

        cv2.rectangle(image,
                      (gt_stixel.column, gt_stixel.top),
                      (gt_stixel.column + stixel_width, gt_stixel.bottom),
                      (238, 58, 140), -1)

        cv2.rectangle(image,
                      (gt_stixel.column, gt_stixel.top),
                      (gt_stixel.column + stixel_width, gt_stixel.bottom),
                      (46, 139, 87), 2)

        if overlap_top < overlap_bottom and overlap_left < overlap_right:
            cv2.rectangle(image,
                          (overlap_left, overlap_top),
                          (overlap_right, overlap_bottom),
                          (0, 205, 102), -1)

        cv2.rectangle(image,
                      (pred_stixel.column, pred_stixel.top),
                      (pred_stixel.column + stixel_width, pred_stixel.bottom),
                      (118, 238, 198), 2)
    return Image.fromarray(image)
