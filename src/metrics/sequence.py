import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List
import numpy as np
from .base import BaseMetric

class BLEU(BaseMetric):
    """BLEU score metric for sequence tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_n = self.config.get('max_n', 4)
        self.weights = self.config.get('weights', [0.25] * 4)  # Equal weights for n-grams 1-4
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: List of predicted sequences
        # targets: List of target sequences
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> float:
        def get_ngrams(sequence, n):
            return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        
        def get_clip_count(pred_ngrams, target_ngrams):
            pred_counts = {}
            for ngram in pred_ngrams:
                pred_counts[ngram] = pred_counts.get(ngram, 0) + 1
            
            clip_count = 0
            for ngram in pred_counts:
                max_count = max([target_ngrams.count(ngram) for target_ngrams in target_ngrams])
                clip_count += min(pred_counts[ngram], max_count)
            
            return clip_count
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            total_clip_count = 0
            total_pred_count = 0
            
            for pred, targets in zip(self.predictions, self.targets):
                pred_ngrams = get_ngrams(pred, n)
                target_ngrams = [get_ngrams(t, n) for t in targets]
                
                clip_count = get_clip_count(pred_ngrams, target_ngrams)
                total_clip_count += clip_count
                total_pred_count += len(pred_ngrams)
            
            precision = total_clip_count / total_pred_count if total_pred_count > 0 else 0
            precisions.append(precision)
        
        # Calculate brevity penalty
        pred_lengths = [len(p) for p in self.predictions]
        target_lengths = [len(t) for t in self.targets]
        bp = 1.0 if min(pred_lengths) > max(target_lengths) else np.exp(1 - max(target_lengths) / min(pred_lengths))
        
        # Calculate BLEU score
        bleu = bp * np.exp(sum(w * np.log(p) for w, p in zip(self.weights, precisions)))
        return bleu

class ROUGE(BaseMetric):
    """ROUGE score metric for sequence tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rouge_type = self.config.get('rouge_type', 'rouge-1')
        self.beta = self.config.get('beta', 1.0)  # F-measure parameter
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        def get_ngrams(sequence, n):
            return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        
        n = int(self.rouge_type.split('-')[1])
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        count = 0
        
        for pred, targets in zip(self.predictions, self.targets):
            pred_ngrams = get_ngrams(pred, n)
            target_ngrams = [get_ngrams(t, n) for t in targets]
            
            # Calculate precision
            pred_counts = {}
            for ngram in pred_ngrams:
                pred_counts[ngram] = pred_counts.get(ngram, 0) + 1
            
            clip_count = 0
            for ngram in pred_counts:
                max_count = max([target_ngrams.count(ngram) for target_ngrams in target_ngrams])
                clip_count += min(pred_counts[ngram], max_count)
            
            precision = clip_count / len(pred_ngrams) if pred_ngrams else 0
            
            # Calculate recall
            target_ngrams_flat = [ngram for target_ngrams in target_ngrams for ngram in target_ngrams]
            target_counts = {}
            for ngram in target_ngrams_flat:
                target_counts[ngram] = target_counts.get(ngram, 0) + 1
            
            recall_count = 0
            for ngram in pred_counts:
                recall_count += min(pred_counts[ngram], target_counts.get(ngram, 0))
            
            recall = recall_count / len(target_ngrams_flat) if target_ngrams_flat else 0
            
            # Calculate F1
            if precision + recall > 0:
                f1 = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall)
            else:
                f1 = 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1
        
        return {
            'precision': total_precision / count if count > 0 else 0,
            'recall': total_recall / count if count > 0 else 0,
            'f1': total_f1 / count if count > 0 else 0
        }

class Perplexity(BaseMetric):
    """Perplexity metric for language modeling tasks."""
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
        # predictions: Log probabilities for each token
        # targets: Target token indices
        loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1), reduction='sum')
        self.total_loss += loss.item()
        self.total_tokens += targets.numel()
    
    def compute(self) -> float:
        return np.exp(self.total_loss / self.total_tokens) if self.total_tokens > 0 else float('inf') 