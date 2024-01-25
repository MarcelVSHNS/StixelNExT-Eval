from metrics.Metric import EvaluationMetric, Stixel
from typing import List
from PIL import Image, ImageDraw
import cv2
import numpy as np


class ObstacleMetric(EvaluationMetric):
    # 1. Implement __init()__
    def __init__(self):
        super().__init__()
        self.scores: List[float] = []

    def evaluate(self, prediction: List[Stixel], target: List[Stixel]):
        minus_points = 0
        max_score = sum(target.bottom for target in target)
        prediction_dict = {pred.column: pred for pred in prediction}

        for target_stixel in target:
            if target_stixel.column in prediction_dict:
                minus_points += abs(prediction_dict[target_stixel.column].bottom - target_stixel.bottom)
        score = max_score - minus_points
        score_percentage = (score / max_score) * 100 if max_score > 0 else 0
        self.scores.append(score_percentage)
        return score_percentage

    def get_score(self):
        return sum(self.scores) / len(self.scores)


def visualize_stixels_on_image(img, predictions, targets, stixel_width=8, name="sampleone"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    prediction_dict = {pred.column: pred for pred in predictions}
    for target in targets:
        if target.column in prediction_dict:
            pred = prediction_dict[target.column]
            color = '#FFC0CB' if pred.bottom < target.bottom else '#30D5C8'  # turquoise '#FFC0CB', pink '#30D5C8'
            start_y = min(pred.bottom, target.bottom)
            end_y = max(pred.bottom, target.bottom)
            draw.rectangle([(target.column, start_y), (target.column + stixel_width, end_y)], fill=color)
        draw.line([(target.column, target.bottom), (target.column + stixel_width, target.bottom)], fill='green', width=4)
    img.save(name + '.png')
