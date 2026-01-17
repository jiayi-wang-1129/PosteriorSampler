"""Telescoping / multilevel posterior sampling (toy 2D examples).

This package refactors the original notebook experiments into importable modules.
See the `examples/` folder for runnable scripts.
"""

from .priors import Prior, GMMPrior, SwissRollPrior
from .likelihood import GaussianParams, GaussianLikelihoodSchedule
from .score_model import MLPScoreNet
from .training import ScoreTrainingConfig, train_score_model, load_checkpoint, save_checkpoint
from .sampler import SamplerConfig, TelescopingPosteriorSampler, SamplingResult

__all__ = [
    "Prior",
    "GMMPrior",
    "SwissRollPrior",
    "GaussianParams",
    "GaussianLikelihoodSchedule",
    "MLPScoreNet",
    "ScoreTrainingConfig",
    "train_score_model",
    "load_checkpoint",
    "save_checkpoint",
    "SamplerConfig",
    "TelescopingPosteriorSampler",
    "SamplingResult",
]
