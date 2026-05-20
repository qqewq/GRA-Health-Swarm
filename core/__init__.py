"""
GRA-Health-Swarm: Swarm Subjectivity for Health, Rejuvenation, and Healing.

Based on:
- P.K. Anokhin's Theory of Functional Systems
- GRA (Gradient Reduction of Argumentative Foam) framework
- Swarm subjectivity contagion
"""

from .agents import (
    HealthAgent,
    AfferentSynthesisAgent,
    DecisionDominanceAgent,
    AcceptorOfActionResultAgent,
    EfferentActionAgent,
    ReverseAfferentationAgent,
    MismatchDetectorAgent,
)
from .health_state import HealthState
from .null_operator import HealthNullOperator, NullificationStep
from .interaction_graph import InteractionGraph, InteractionEdge, EnvironmentNode
from .rejuvenation_loop import RejuvenationLoop, RejuvenationMetrics

__version__ = "0.1.0"
__author__ = "GRA Health Swarm Team"

__all__ = [
    "HealthAgent",
    "AfferentSynthesisAgent",
    "DecisionDominanceAgent",
    "AcceptorOfActionResultAgent",
    "EfferentActionAgent",
    "ReverseAfferentationAgent",
    "MismatchDetectorAgent",
    "HealthState",
    "HealthNullOperator",
    "NullificationStep",
    "InteractionGraph",
    "InteractionEdge",
    "EnvironmentNode",
    "RejuvenationLoop",
    "RejuvenationMetrics",
]