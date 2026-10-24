"""Módulo de identificação e classificação de times dos jogadores."""
from src.teams.classifier import TeamClassifier, TeamVotingManager
from src.teams.features import SigLIPEmbedder, extract_hsv_features

__all__ = [
    "TeamClassifier",
    "TeamVotingManager",
    "extract_hsv_features",
    "SigLIPEmbedder",
]
