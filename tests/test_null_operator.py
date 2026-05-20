"""Tests for health nullification operator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core import (
    AfferentSynthesisAgent,
    DecisionDominanceAgent,
    AcceptorOfActionResultAgent,
    EfferentActionAgent,
    ReverseAfferentationAgent,
    MismatchDetectorAgent,
    HealthState,
    HealthNullOperator
)


def create_test_swarm():
    """Create a test swarm of health agents."""
    agents = {
        "afferent_synth": AfferentSynthesisAgent("afferent_synth", 5),
        "decision_dom": DecisionDominanceAgent("decision_dom", 5),
        "aar_main": AcceptorOfActionResultAgent("aar_main", 5),
        "efferent_motor": EfferentActionAgent("efferent_motor", 5),
        "reverse_afferent": ReverseAfferentationAgent("reverse_afferent", 5),
        "mismatch_detect": MismatchDetectorAgent("mismatch_detect", 1),
    }

    # Introduce mismatch: set states far from predictions
    for agent in agents.values():
        agent.state = np.random.normal(0, 1, agent.state.shape[0])
        agent.prediction = np.zeros(agent.state.shape[0])
        agent.compute_foam()

    return agents


def test_null_operator_reduces_foam():
    agents = create_test_swarm()
    health_state = HealthState()
    
    # Compute initial foam
    agent_foams = {aid: a.compute_foam() for aid, a in agents.items()}
    health_state.compute_total_foam(agent_foams)
    phi_before = health_state.phi_total

    # Apply nullification
    null_op = HealthNullOperator(learning_rate=0.3)
    agents, health_state = null_op.apply(agents, health_state)
    phi_after = health_state.phi_total

    print(f"Φ before: {phi_before:.4f}")
    print(f"Φ after:  {phi_after:.4f}")
    assert phi_after <= phi_before or abs(phi_after - phi_before) < 1e-6


def test_chronic_stress_detection():
    agents = create_test_swarm()
    null_op = HealthNullOperator()

    # Simulate chronic stress
    detector: MismatchDetectorAgent = agents["mismatch_detect"]
    detector.chronic_stress_counter = 100
    assert null_op._is_chronic_stress(agents)


def test_convergence():
    agents = create_test_swarm()
    health_state = HealthState()
    null_op = HealthNullOperator(learning_rate=0.2)

    phi_history = []
    for _ in range(20):
        agent_foams = {aid: a.compute_foam() for aid, a in agents.items()}
        health_state.compute_total_foam(agent_foams)
        phi_history.append(health_state.phi_total)
        agents, health_state = null_op.apply(agents, health_state)

    # Check that foam generally decreases
    assert phi_history[-1] <= phi_history[0] or abs(phi_history[-1] - phi_history[0]) < 0.1


if __name__ == "__main__":
    test_null_operator_reduces_foam()
    test_chronic_stress_detection()
    test_convergence()
    print("All null operator tests passed! ✓")