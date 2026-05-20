"""
Abstract health/stress state representation.

Defines the HealthState object that aggregates all agents' states
and computes total foam Φ_total for the organism.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class HealthState:
    """Aggregated health state of the organism (swarm of functional agents)."""
    # Core metrics
    phi_total: float = 0.0  # Total foam Φ_total
    phi_internal: float = 0.0  # Sum of individual agent foams ΣΦ(Ψᵢ)
    phi_cross: float = 0.0  # Sum of cross-agent foams ΣΦ_cross(Ψᵢ, Ψⱼ)

    # Health indicators
    health_index: float = 1.0  # 1.0 = optimal health, 0.0 = critical
    stress_level: float = 0.0  # Acute stress
    chronic_stress_risk: float = 0.0  # Risk of chronic stress → disease

    # Rejuvenation metrics
    rejuvenation_potential: float = 0.0  # How much Φ can still be reduced
    aar_accuracy: float = 0.0  # How accurate are predictions
    adaptability_index: float = 0.5  # System's ability to recalibrate

    # Time series
    phi_history: List[float] = field(default_factory=list)
    health_history: List[float] = field(default_factory=list)

    # Agent states
    agent_states: Dict[str, np.ndarray] = field(default_factory=dict)
    agent_foams: Dict[str, float] = field(default_factory=dict)

    def compute_total_foam(self, agent_foams: Dict[str, float],
                          cross_foams: Optional[Dict[tuple, float]] = None) -> float:
        """Compute Φ_total from individual and cross-agent foams."""
        self.phi_internal = sum(agent_foams.values())
        self.agent_foams = agent_foams

        if cross_foams:
            self.phi_cross = sum(cross_foams.values())
        else:
            self.phi_cross = 0.0

        self.phi_total = self.phi_internal + self.phi_cross
        self.phi_history.append(self.phi_total)

        # Update health index
        self.health_index = max(0.0, 1.0 - self.phi_total / 10.0)

        return self.phi_total

    def is_healthy(self, threshold: float = 0.3) -> bool:
        """Check if system is in healthy state (low total foam)."""
        return self.phi_total < threshold

    def is_diseased(self, threshold: float = 2.0) -> bool:
        """Check if system is in disease state (chronically high foam)."""
        return self.phi_total > threshold and len(self.phi_history) > 10 and \
               all(phi > threshold for phi in self.phi_history[-10:])

    def is_rejuvenating(self) -> bool:
        """Check if system is actively rejuvenating (Φ decreasing)."""
        if len(self.phi_history) < 5:
            return False
        recent = self.phi_history[-5:]
        return all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

    def get_trend(self) -> str:
        """Get health trend: 'improving', 'declining', or 'stable'."""
        if len(self.phi_history) < 10:
            return "unknown"
        first_half = np.mean(self.phi_history[:5])
        second_half = np.mean(self.phi_history[-5:])
        if second_half < first_half - 0.1:
            return "improving"
        elif second_half > first_half + 0.1:
            return "declining"
        else:
            return "stable"