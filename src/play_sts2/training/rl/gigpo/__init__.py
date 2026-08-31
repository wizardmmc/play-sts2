"""提供整局 GiGPO episode、精确 anchor census 与训练投影。"""

from .collector import BackboneGroupCollector
from .contracts import (
    GigpoAnchorCensus,
    GigpoEpisode,
    GigpoGroup,
    GigpoGroupRejected,
    GigpoStep,
    build_gigpo_group,
)
from .entrypoint import collect_gigpo_group
from .io import write_gigpo_group
from .learner import load_gigpo_training_group
from .rollout import (
    BackboneCheckpointCandidate,
    BackboneEpisodeDraft,
    build_gigpo_episode,
)
from .scenario import BackboneBattleCandidate, extract_battle_candidate
from .selection import select_battle_candidates
from .worker import GameBackboneWorker

__all__ = [
    "BackboneBattleCandidate",
    "BackboneCheckpointCandidate",
    "BackboneEpisodeDraft",
    "BackboneGroupCollector",
    "GameBackboneWorker",
    "GigpoAnchorCensus",
    "GigpoEpisode",
    "GigpoGroup",
    "GigpoGroupRejected",
    "GigpoStep",
    "build_gigpo_episode",
    "build_gigpo_group",
    "collect_gigpo_group",
    "extract_battle_candidate",
    "load_gigpo_training_group",
    "select_battle_candidates",
    "write_gigpo_group",
]
