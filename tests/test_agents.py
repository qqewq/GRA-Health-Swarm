"""Tests for health agents."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.agents import (
    AfferentSynthesisAgent,
    DecisionDominanceAgent,
    AcceptorOfActionResultAgent,
    EfferentActionAgent,
    ReverseAfferentationAgent,
    MismatchDetectorAgent
)


def test_afferent_synthesis():
    agent = AfferentSynthesisAgent("test_aff", dim=5)
    ext = np.ones(5)
    int_state = np.zeros(5)
    mem = np.ones(5) * 0.5
    result = agent.synthesize(ext, int_state, mem)
    assert result.shape == (5,)
    assert agent.foam >= 0


def test_decision_dominance():
    agent = DecisionDominanceAgent("test_dec", dim=5)
    info = np.ones(5)
    motivation = np.ones(5) * 0.8
    goal = agent.form_dominant_goal(info, motivation)
    assert goal.shape == (5,)
    assert not agent.is_rigid()  # Default adaptability is 0.5


def test_aar():
    agent = AcceptorOfActionResultAgent("test_aar", dim=5)
    goal = np.ones(5)
    past = np.zeros(5)
    prediction = agent.build_acceptor(goal, past)
    assert prediction.shape == (5,)
    mismatch, matched = agent.compare_with_reality(np.ones(5))
    assert mismatch >= 0
    assert matched  # Same vector should match


def test_efferent_action():
    agent = EfferentActionAgent("test_eff", dim=5)
    command = np.ones(5)
    result = agent.execute(command, noise=0.0)
    assert np.allclose(result, command)


def test_reverse_afferentation():
    agent = ReverseAfferentationAgent("test_rev", dim=5)
    body = np.ones(5)
    env = np.zeros(5)
    feedback = agent.collect_feedback(body, env)
    assert feedback.shape == (5,)


def test_mismatch_detector():
    agent = MismatchDetectorAgent("test_mis", dim=1)
    aar = np.ones(5)
    feedback = np.ones(5) * 0.5
    mismatch = agent.detect_mismatch(aar, feedback)
    assert mismatch > 0
    assert not agent.is_chronic_stress(threshold=100)


if __name__ == "__main__":
    test_afferent_synthesis()
    test_decision_dominance()
    test_aar()
    test_efferent_action()
    test_reverse_afferentation()
    test_mismatch_detector()
    print("All agent tests passed! ✓")