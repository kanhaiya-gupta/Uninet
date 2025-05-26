"""Metrics for neural network evaluation."""

from .base import BaseMetric
from .classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    ROCAUC,
    PRCurve
)
from .regression import (
    MSE,
    MAE,
    RMSE,
    R2Score,
    ExplainedVariance
)
from .visualization import (
    PSNR,
    SSIM,
    LPIPS,
    LossPlotter,
    MetricPlotter
)
from .graph import (
    NodeClassificationAccuracy,
    GraphClassificationAccuracy,
    EdgePredictionMetrics,
    GraphSimilarityMetrics
)
from .sequence import (
    Perplexity,
    BLEU,
    ROUGE,
    SequenceAccuracy
)
from .detection import (
    IoU,
    mAP,
    PrecisionRecallCurve
)
from .segmentation import (
    DiceScore,
    JaccardIndex,
    PixelAccuracy
)
from .generative import (
    FID,
    IS,
    KID
)
from .recommendation import (
    HitRate,
    NDCG,
    MRR
)
from .physics import (
    ConservationError,
    StabilityError
)
from .function import (
    ApproximationError,
    InterpolationError
)
from .clustering import (
    SilhouetteScore,
    CalinskiHarabaszScore
)
from .feature import (
    FeatureImportance,
    FeatureCorrelation
)
from .dynamics import (
    TrajectoryError,
    PhaseSpaceError
)
from .spiking import (
    SpikeRate,
    SpikeTiming
)

__all__ = [
    'BaseMetric',
    # Classification metrics
    'Accuracy',
    'Precision',
    'Recall',
    'F1Score',
    'ConfusionMatrix',
    'ROCAUC',
    'PRCurve',
    # Regression metrics
    'MSE',
    'MAE',
    'RMSE',
    'R2Score',
    'ExplainedVariance',
    # Visualization metrics
    'PSNR',
    'SSIM',
    'LPIPS',
    'LossPlotter',
    'MetricPlotter',
    # Graph metrics
    'NodeClassificationAccuracy',
    'GraphClassificationAccuracy',
    'EdgePredictionMetrics',
    'GraphSimilarityMetrics',
    # Sequence metrics
    'Perplexity',
    'BLEU',
    'ROUGE',
    'SequenceAccuracy',
    # Detection metrics
    'IoU',
    'mAP',
    'PrecisionRecallCurve',
    # Segmentation metrics
    'DiceScore',
    'JaccardIndex',
    'PixelAccuracy',
    # Generative metrics
    'FID',
    'IS',
    'KID',
    # Recommendation metrics
    'HitRate',
    'NDCG',
    'MRR',
    # Physics metrics
    'ConservationError',
    'StabilityError',
    # Function metrics
    'ApproximationError',
    'InterpolationError',
    # Clustering metrics
    'SilhouetteScore',
    'CalinskiHarabaszScore',
    # Feature metrics
    'FeatureImportance',
    'FeatureCorrelation',
    # Dynamics metrics
    'TrajectoryError',
    'PhaseSpaceError',
    # Spiking metrics
    'SpikeRate',
    'SpikeTiming'
] 