"""
Health-adapted GRA Nullification Operator (N_health).

This operator detects high-foam configurations in a health swarm
and applies transformations to drive Φ_total → 0.

Based on Anokhin's principle: recalibrate the AAR when mismatch is detected.
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass

from .agents import (
    HealthAgent,
    AfferentSynthesisAgent,
    DecisionDominanceAgent,
    AcceptorOfActionResultAgent,
    EfferentActionAgent,
    ReverseAfferentationAgent,
    MismatchDetectorAgent
)
from .health_state import HealthState


@dataclass
class NullificationStep:
    """Record of one nullification step."""
    step_id: int
    phi_before: float
    phi_after: float
    action: str  # What was done
    agents_affected: List[str]
    success: bool


class HealthNullOperator:
    """
    N_health: GRA-reset operator specialized for health and healing.

    Core mechanism:
    1. Detect high Φ configurations
    2. Identify source of mismatch (which agent's AAR is wrong?)
    3. Recalibrate predictions (AAR adjustment)
    4. Strengthen feedback loops (reverse afferentation)
    5. If needed, form compensatory functional systems
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 recalibration_threshold: float = 0.5,
                 chronic_threshold: int = 50):
        self.learning_rate = learning_rate
        self.recalibration_threshold = recalibration_threshold
        self.chronic_threshold = chronic_threshold
        self.history: List[NullificationStep] = []
        self.step_counter = 0

    def apply(self,
              agents: Dict[str, HealthAgent],
              health_state: HealthState,
              target_state: Optional[np.ndarray] = None) -> Tuple[Dict[str, HealthAgent], HealthState]:
        """
        Apply one step of N_health to the swarm.

        Args:
            agents: Dictionary of functional agents
            health_state: Current health state
            target_state: Desired target state (if known)

        Returns:
            Updated agents and health state
        """
        phi_before = health_state.phi_total

        # 1. Detect which agents have high foam
        high_foam_agents = self._detect_high_foam_agents(agents)

        # 2. For each high-foam agent, determine the action
        affected_agents = []
        for agent_id in high_foam_agents:
            agent = agents[agent_id]
            action_taken = self._apply_correction(agent, agents, target_state)
            affected_agents.append(agent_id)

        # 3. If chronic stress detected, apply deeper recalibration
        if self._is_chronic_stress(agents):
            self._apply_chronic_recalibration(agents)

        # 4. Strengthen cross-agent coherence
        self._reduce_cross_foam(agents)

        # 5. Recompute health state
        agent_foams = {aid: a.compute_foam() for aid, a in agents.items()}
        cross_foams = self._compute_cross_foams(agents)
        health_state.compute_total_foam(agent_foams, cross_foams)

        phi_after = health_state.phi_total

        # Record step
        step = NullificationStep(
            step_id=self.step_counter,
            phi_before=phi_before,
            phi_after=phi_after,
            action="multi-agent correction",
            agents_affected=affected_agents,
            success=phi_after < phi_before
        )
        self.history.append(step)
        self.step_counter += 1

        return agents, health_state

    def _detect_high_foam_agents(self, agents: Dict[str, HealthAgent]) -> List[str]:
        """Find agents with foam above threshold."""
        high_foam = []
        for agent_id, agent in agents.items():
            if agent.foam > self.recalibration_threshold:
                high_foam.append(agent_id)
        return high_foam

    def _apply_correction(self,
                         agent: HealthAgent,
                         all_agents: Dict[str, HealthAgent],
                         target_state: Optional[np.ndarray] = None) -> str:
        """Apply appropriate correction based on agent type."""
        if agent.agent_type == "aar":
            # Recalibrate AAR — the core healing mechanism
            agent.adjust_prediction(learning_rate=self.learning_rate)
            return "aar_recalibration"
        elif agent.agent_type == "mismatch_detector":
            # Reduce sensitivity temporarily to break chronic stress loop
            agent.stress_level *= 0.9
            return "stress_reduction"
        elif agent.agent_type == "decision_dominance":
            # Increase adaptability if dominance is too rigid
            if agent.is_rigid():
                agent.adaptability = min(1.0, agent.adaptability + 0.05)
            return "dominance_flexibilization"
        elif agent.agent_type == "reverse_afferentation":
            # Strengthen feedback — collect more accurate body state
            return "feedback_strengthening"
        else:
            # General adjustment
            agent.adaptability = min(1.0, agent.adaptability + 0.02)
            return "general_adaptation"

    def _is_chronic_stress(self, agents: Dict[str, HealthAgent]) -> bool:
        """Check if any mismatch detector reports chronic stress."""
        for agent in agents.values():
            if agent.agent_type == "mismatch_detector":
                detector: MismatchDetectorAgent = agent
                return detector.is_chronic_stress(self.chronic_threshold)
        return False

    def _apply_chronic_recalibration(self, agents: Dict[str, HealthAgent]):
        """Deep recalibration for chronic stress: reset all AARs."""
        for agent in agents.values():
            if agent.agent_type == "aar":
                # Strong recalibration
                agent.adjust_prediction(learning_rate=0.3)
            # Increase global adaptability
            agent.adaptability = min(1.0, agent.adaptability + 0.03)

    def _reduce_cross_foam(self, agents: Dict[str, HealthAgent]):
        """Reduce cross-agent foam by aligning predictions."""
        aar_agents = [a for a in agents.values() if a.agent_type == "aar"]
        if len(aar_agents) >= 2:
            # Align all AARs toward their mean
            predictions = [a.prediction for a in aar_agents if a.prediction is not None]
            if predictions:
                mean_pred = np.mean(predictions, axis=0)
                for a in aar_agents:
                    if a.prediction is not None:
                        a.prediction = 0.8 * a.prediction + 0.2 * mean_pred

    def _compute_cross_foams(self, agents: Dict[str, HealthAgent]) -> Dict[tuple, float]:
        """Compute cross-agent foams (mismatch between connected agents)."""
        cross_foams = {}
        agent_list = list(agents.values())
        for i, a1 in enumerate(agent_list):
            for j, a2 in enumerate(agent_list):
                if i < j and a1.agent_type != a2.agent_type:
                    # Cross-foam based on state divergence
                    cross = np.linalg.norm(a1.state - a2.state)
                    cross_foams[(a1.agent_id, a2.agent_id)] = cross
        return cross_foams

    def get_nullification_history(self) -> List[NullificationStep]:
        """Get the history of nullification steps."""
        return self.history

    def is_converged(self, tolerance: float = 0.01) -> bool:
        """Check if nullification has converged (Φ is stable near zero)."""
        if len(self.history) < 5:
            return False
        recent_phis = [step.phi_after for step in self.history[-5:]]
        return all(phi < tolerance for phi in recent_phis)