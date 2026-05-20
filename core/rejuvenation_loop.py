"""
Rejuvenation Loop: Core algorithm for healing and rejuvenation.

Implements the iterative process where:
1. AAR is recalibrated based on mismatch feedback
2. Φ_total is driven toward zero
3. Compensatory functional systems are activated when needed
4. The system moves toward optimal health (rejuvenation)

Based on Anokhin's principle: the organism can restore function
through dynamic reorganization of functional systems.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

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
from .null_operator import HealthNullOperator
from .interaction_graph import InteractionGraph


@dataclass
class RejuvenationMetrics:
    """Metrics tracked during the rejuvenation process."""
    phi_trajectory: List[float] = field(default_factory=list)
    health_trajectory: List[float] = field(default_factory=list)
    aar_accuracy_trajectory: List[float] = field(default_factory=list)
    stress_trajectory: List[float] = field(default_factory=list)
    rejuvenation_speed: float = 0.0  # Rate of Φ reduction
    total_steps_to_health: Optional[int] = None
    relapses: int = 0  # Number of times Φ increased again


class RejuvenationLoop:
    """
    Main rejuvenation loop: iterative healing through GRA nullification.

    Algorithm:
    1. Initialize functional system agents
    2. For each step:
       a. Collect afferent synthesis (environment + internal state)
       b. Form dominant goal (decision)
       c. Build AAR (prediction)
       d. Execute action (efferent)
       e. Collect reverse afferentation (feedback)
       f. Detect mismatch (Φ computation)
       g. Apply N_health operator (recalibrate if Φ is high)
    3. Track Φ_total trajectory toward zero
    """

    def __init__(self, dim: int = 10, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate

        # Initialize agents
        self.agents: Dict[str, HealthAgent] = {}
        self._init_agents()

        # Initialize components
        self.health_state = HealthState()
        self.null_operator = HealthNullOperator(learning_rate=learning_rate)
        self.interaction_graph = InteractionGraph()
        self.interaction_graph.build_default_anokhin_graph(dim)

        # Metrics
        self.metrics = RejuvenationMetrics()

        # Target (healthy) state
        self.target_state = np.ones(dim) * 0.1  # Low-foam target

    def _init_agents(self):
        """Initialize all functional system agents."""
        self.agents["afferent_synth"] = AfferentSynthesisAgent("afferent_synth", self.dim)
        self.agents["decision_dom"] = DecisionDominanceAgent("decision_dom", self.dim)
        self.agents["aar_main"] = AcceptorOfActionResultAgent("aar_main", self.dim)
        self.agents["efferent_motor"] = EfferentActionAgent("efferent_motor", self.dim)
        self.agents["efferent_visceral"] = EfferentActionAgent("efferent_visceral", self.dim)
        self.agents["reverse_afferent"] = ReverseAfferentationAgent("reverse_afferent", self.dim)
        self.agents["mismatch_detect"] = MismatchDetectorAgent("mismatch_detect", 1)
        self.agents["aar_backup"] = AcceptorOfActionResultAgent("aar_backup", self.dim)

    def step(self,
             external_stimuli: np.ndarray,
             internal_state: np.ndarray,
             environment_stress: float = 0.0,
             social_support: float = 0.0) -> RejuvenationMetrics:
        """
        Execute one step of the rejuvenation loop.

        Args:
            external_stimuli: Environmental input
            internal_state: Current body state
            environment_stress: External stress level (e.g., drone threat)
            social_support: Available social support

        Returns:
            Updated metrics
        """
        # 1. Afferent synthesis
        afferent: AfferentSynthesisAgent = self.agents["afferent_synth"]
        memory_context = np.mean(afferent.memory, axis=0) if afferent.memory else np.zeros(self.dim)
        synthesized = afferent.synthesize(external_stimuli, internal_state, memory_context)

        # 2. Decision / dominance
        decision: DecisionDominanceAgent = self.agents["decision_dom"]
        motivation = self.target_state - internal_state  # Drive toward health
        goal = decision.form_dominant_goal(synthesized, motivation)

        # 3. Build AAR
        aar: AcceptorOfActionResultAgent = self.agents["aar_main"]
        past_exp = np.mean(aar.memory, axis=0) if aar.memory else np.zeros(self.dim)
        prediction = aar.build_acceptor(goal, past_exp)

        # Also update backup AAR
        aar_backup: AcceptorOfActionResultAgent = self.agents["aar_backup"]
        aar_backup.build_acceptor(goal, past_exp)

        # 4. Execute action
        efferent: EfferentActionAgent = self.agents["efferent_motor"]
        action_result = efferent.execute(prediction)

        # 5. Reverse afferentation (feedback)
        feedback_agent: ReverseAfferentationAgent = self.agents["reverse_afferent"]
        body_state = internal_state + 0.3 * action_result  # Body responds to action
        env_state = external_stimuli * (1.0 - environment_stress)  # Environment modified by stress
        feedback = feedback_agent.collect_feedback(body_state, env_state)

        # 6. Mismatch detection
        detector: MismatchDetectorAgent = self.agents["mismatch_detect"]
        mismatch = detector.detect_mismatch(prediction, feedback)

        # 7. Apply N_health nullification
        # First compute current health state
        agent_foams = {aid: a.compute_foam() for aid, a in self.agents.items()}
        self.health_state.compute_total_foam(agent_foams)

        # Apply nullification
        self.agents, self.health_state = self.null_operator.apply(
            self.agents, self.health_state, self.target_state
        )

        # Track metrics
        self.metrics.phi_trajectory.append(self.health_state.phi_total)
        self.metrics.health_trajectory.append(self.health_state.health_index)
        self.metrics.aar_accuracy_trajectory.append(
            1.0 / (1.0 + mismatch)  # Higher accuracy = lower mismatch
        )
        self.metrics.stress_trajectory.append(detector.stress_level)

        # Check for relapse
        if len(self.metrics.phi_trajectory) >= 2:
            if self.metrics.phi_trajectory[-1] > self.metrics.phi_trajectory[-2]:
                self.metrics.relapses += 1

        # Compute rejuvenation speed
        if len(self.metrics.phi_trajectory) >= 10:
            recent = self.metrics.phi_trajectory[-10:]
            self.metrics.rejuvenation_speed = (recent[0] - recent[-1]) / 10.0

        return self.metrics

    def run_until_healthy(self,
                         max_steps: int = 1000,
                         health_threshold: float = 0.1,
                         external_stimuli_generator=None,
                         internal_state_generator=None,
                         stress_generator=None,
                         support_generator=None) -> RejuvenationMetrics:
        """
        Run rejuvenation loop until Φ_total is below threshold or max steps reached.

        Returns metrics when done.
        """
        for step_idx in range(max_steps):
            # Generate inputs (default: random fluctuations around target)
            if external_stimuli_generator:
                ext = external_stimuli_generator(step_idx)
            else:
                ext = self.target_state + np.random.normal(0, 0.05, self.dim)

            if internal_state_generator:
                int_state = internal_state_generator(step_idx)
            else:
                int_state = self.target_state + np.random.normal(0, 0.1, self.dim)

            stress = stress_generator(step_idx) if stress_generator else 0.0
            support = support_generator(step_idx) if support_generator else 0.5

            self.step(ext, int_state, stress, support)

            # Check convergence
            if self.health_state.phi_total < health_threshold:
                self.metrics.total_steps_to_health = step_idx + 1
                break

        return self.metrics

    def get_current_state_summary(self) -> dict:
        """Get a human-readable summary of current health state."""
        return {
            "phi_total": self.health_state.phi_total,
            "health_index": self.health_state.health_index,
            "stress_level": self.agents["mismatch_detect"].stress_level,
            "is_healthy": self.health_state.is_healthy(),
            "is_diseased": self.health_state.is_diseased(),
            "is_rejuvenating": self.health_state.is_rejuvenating(),
            "trend": self.health_state.get_trend(),
            "aar_accuracy": self.metrics.aar_accuracy_trajectory[-1] if self.metrics.aar_accuracy_trajectory else 0.0,
            "chronic_stress_risk": self.agents["mismatch_detect"].is_chronic_stress(),
            "rejuvenation_speed": self.metrics.rejuvenation_speed,
            "dominance_rigidity": self.agents["decision_dom"].is_rigid(),
        }