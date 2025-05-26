import torch
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from .base import BaseMetric

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    # Get coordinates of intersection rectangle
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    
    # Compute intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute areas of both boxes
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Compute union area
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-8)

class MeanAveragePrecision(BaseMetric):
    """Mean Average Precision metric for object detection tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.iou_thresholds = self.config.get('iou_thresholds', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        self.num_classes = self.config.get('num_classes', 80)
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: List of (boxes, scores, labels) for each image
        # targets: List of (boxes, labels) for each image
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        aps = []
        for iou_threshold in self.iou_thresholds:
            ap = self._compute_ap(iou_threshold)
            aps.append(ap)
        
        return {
            'mAP': np.mean(aps),
            'AP_per_threshold': {f'IoU_{t:.2f}': ap for t, ap in zip(self.iou_thresholds, aps)}
        }
    
    def _compute_ap(self, iou_threshold: float) -> float:
        # Sort predictions by confidence score
        all_predictions = []
        for pred in self.predictions:
            boxes, scores, labels = pred
            for box, score, label in zip(boxes, scores, labels):
                all_predictions.append((box, score, label))
        
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize variables
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        gt_matched = set()
        
        # Match predictions to ground truth
        for i, (pred_box, _, pred_label) in enumerate(all_predictions):
            matched = False
            for target in self.targets:
                target_boxes, target_labels = target
                for j, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                    if target_label == pred_label and j not in gt_matched:
                        iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0))
                        if iou > iou_threshold:
                            tp[i] = 1
                            gt_matched.add(j)
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (len(self.targets) + 1e-8)
        
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        return ap

class PrecisionAtK(BaseMetric):
    """Precision at K metric for object detection tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.k = self.config.get('k', 100)
        self.iou_threshold = self.config.get('iou_threshold', 0.5)
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> float:
        # Sort predictions by confidence score
        all_predictions = []
        for pred in self.predictions:
            boxes, scores, labels = pred
            for box, score, label in zip(boxes, scores, labels):
                all_predictions.append((box, score, label))
        
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        all_predictions = all_predictions[:self.k]
        
        # Match predictions to ground truth
        tp = 0
        for pred_box, _, pred_label in all_predictions:
            for target in self.targets:
                target_boxes, target_labels = target
                for target_box, target_label in zip(target_boxes, target_labels):
                    if target_label == pred_label:
                        iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0))
                        if iou > self.iou_threshold:
                            tp += 1
                            break
        
        return tp / len(all_predictions) if all_predictions else 0.0 