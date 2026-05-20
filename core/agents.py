"""
Health-related functional agents based on P.K. Anokhin's Theory of Functional Systems.

Each agent represents a component of a functional system:
- Afferent synthesis
- Decision-making / dominance
- Acceptor of action result (AAR)
- Efferent action
- Reverse afferentation
- Mismatch detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class HealthAgent:
    """Base agent representing a functional subsystem of the organism."""
    agent_id: str
    agent_type: str  # e.g., "afferent_synthesis", "aar", "efferent", etc.
    state: np.ndarray  # Current state vector Ψ
    prediction: Optional[np.ndarray] = None  # Predicted result (AAR)
    foam: float = 0.0  # Φ(Ψ) — internal inconsistency
    memory: List[np.ndarray] = field(default_factory=list)  # Past experience
    connections: List[str] = field(default_factory=list)  # Connected agent IDs
    dominance: float = 0.0  # Current dominance level
    adaptability: float = 0.5  # How flexibly agent can adjust predictions
    health_index: float = 1.0  # 1.0 = optimal, 0.0 = failure

    def compute_foam(self) -> float:
        """Compute internal foam Φ based on prediction-reality mismatch."""
        if self.prediction is not None and len(self.state) > 0:
            mismatch = np.linalg.norm(self.state - self.prediction)
            self.foam = mismatch / (np.linalg.norm(self.state) + 1e-8)
        else:
            self.foam = 0.0
        return self.foam

    def update_state(self, new_state: np.ndarray):
        """Update agent state with new sensory/feedback information."""
        self.memory.append(self.state.copy())
        if len(self.memory) > 100:  # Keep limited memory
            self.memory.pop(0)
        self.state = new_state
        self.compute_foam()

    def form_prediction(self, target_state: np.ndarray):
        """Form AAR — prediction of future result."""
        # Blend target with past experience
        if self.memory:
            past_avg = np.mean(self.memory[-10:], axis=0)
            self.prediction = 0.7 * target_state + 0.3 * past_avg
        else:
            self.prediction = target_state

    def adjust_prediction(self, learning_rate: float = 0.1):
        """Recalibrate AAR based on mismatch (adaptability mechanism)."""
        if self.prediction is not None and self.memory:
            # Move prediction closer to recent reality
            recent_reality = np.mean(self.memory[-5:], axis=0)
            self.prediction = (1 - learning_rate * self.adaptability) * self.prediction + \
                            learning_rate * self.adaptability * recent_reality


class AfferentSynthesisAgent(HealthAgent):
    """Agent responsible for gathering and integrating sensory information."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="afferent_synthesis",
            state=np.zeros(dim)
        )

    def synthesize(self,
                   external_stimuli: np.ndarray,
                   internal_state: np.ndarray,
                   memory_context: np.ndarray) -> np.ndarray:
        """Perform afferent synthesis: combine external, internal, and memory info."""
        synthesized = 0.4 * external_stimuli + 0.3 * internal_state + 0.3 * memory_context
        self.update_state(synthesized)
        return synthesized


class DecisionDominanceAgent(HealthAgent):
    """Agent that forms the dominant goal and makes decisions."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="decision_dominance",
            state=np.zeros(dim)
        )
        self.dominance = 0.8  # High initial dominance

    def form_dominant_goal(self,
                           synthesized_info: np.ndarray,
                           motivation: np.ndarray) -> np.ndarray:
        """Form the dominant motivational goal from synthesized information."""
        goal = 0.6 * motivation + 0.4 * synthesized_info
        self.update_state(goal)
        return goal

    def is_rigid(self) -> bool:
        """Check if dominance is too rigid (low adaptability → chronic stress risk)."""
        return self.adaptability < 0.3


class AcceptorOfActionResultAgent(HealthAgent):
    """Agent that maintains the AAR — prediction of the future result."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="aar",
            state=np.zeros(dim)
        )

    def build_acceptor(self, goal: np.ndarray, past_experience: np.ndarray) -> np.ndarray:
        """Build the acceptor of action result (prediction model)."""
        self.prediction = 0.7 * goal + 0.3 * past_experience
        return self.prediction

    def compare_with_reality(self, real_result: np.ndarray) -> Tuple[float, bool]:
        """Compare prediction with real outcome. Returns (mismatch, is_matched)."""
        if self.prediction is None:
            return 0.0, True
        mismatch = np.linalg.norm(real_result - self.prediction)
        threshold = 0.1 * np.linalg.norm(self.prediction)
        is_matched = mismatch < threshold
        return mismatch, is_matched


class EfferentActionAgent(HealthAgent):
    """Agent that executes motor/physiological commands."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="efferent_action",
            state=np.zeros(dim)
        )

    def execute(self, command: np.ndarray, noise: float = 0.05) -> np.ndarray:
        """Execute action with some physiological noise."""
        noisy_command = command + np.random.normal(0, noise, size=command.shape)
        self.update_state(noisy_command)
        return noisy_command


class ReverseAfferentationAgent(HealthAgent):
    """Agent that collects feedback from body and environment."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="reverse_afferentation",
            state=np.zeros(dim)
        )

    def collect_feedback(self,
                         body_state: np.ndarray,
                         environment_state: np.ndarray) -> np.ndarray:
        """Collect reverse afferentation — feedback about action results."""
        feedback = 0.6 * body_state + 0.4 * environment_state
        self.update_state(feedback)
        return feedback


class MismatchDetectorAgent(HealthAgent):
    """Agent that detects mismatch between AAR and reality — the stress signal."""
    def __init__(self, agent_id: str, dim: int = 10):
        super().__init__(
            agent_id=agent_id,
            agent_type="mismatch_detector",
            state=np.zeros(dim)
        )
        self.stress_level = 0.0
        self.chronic_stress_counter = 0

    def detect_mismatch(self, aar: np.ndarray, feedback: np.ndarray) -> float:
        """Detect mismatch → stress signal."""
        mismatch = np.linalg.norm(feedback - aar)
        self.stress_level = mismatch
        self.update_state(np.array([mismatch]))

        if mismatch > 0.5:
            self.chronic_stress_counter += 1
        else:
            self.chronic_stress_counter = max(0, self.chronic_stress_counter - 1)

        return mismatch

    def is_chronic_stress(self, threshold: int = 50) -> bool:
        """Check if mismatch has been persistent (chronic stress → disease risk)."""
        return self.chronic_stress_counter > threshold