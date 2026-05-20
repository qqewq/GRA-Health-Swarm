"""
Interaction graph: brain–body–environment connectivity.

Models how functional agents interact with each other and with
external factors (environment, stressors, social support, etc.).
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class InteractionEdge:
    """Edge in the interaction graph."""
    source: str
    target: str
    weight: float  # Connection strength
    interaction_type: str  # "excitatory", "inhibitory", "feedback", "prediction"
    delay: int = 0  # Time delay in steps


@dataclass
class EnvironmentNode:
    """External environmental factor."""
    node_id: str
    node_type: str  # "stressor", "support", "nutrition", "social", etc.
    intensity: float  # Current intensity
    predictability: float  # How predictable this factor is


class InteractionGraph:
    """
    Graph representing all interactions in the health swarm system.

    Nodes: agents + environment factors
    Edges: functional connections (predictions, feedback, stress, support)
    """

    def __init__(self):
        self.agent_nodes: Dict[str, str] = {}  # agent_id → agent_type
        self.env_nodes: Dict[str, EnvironmentNode] = {}
        self.edges: List[InteractionEdge] = []
        self.adjacency: Dict[str, List[str]] = {}

    def add_agent_node(self, agent_id: str, agent_type: str):
        """Add an agent to the graph."""
        self.agent_nodes[agent_id] = agent_type
        if agent_id not in self.adjacency:
            self.adjacency[agent_id] = []

    def add_env_node(self, env_node: EnvironmentNode):
        """Add an environmental factor."""
        self.env_nodes[env_node.node_id] = env_node
        if env_node.node_id not in self.adjacency:
            self.adjacency[env_node.node_id] = []

    def add_edge(self, source: str, target: str, weight: float,
                 interaction_type: str, delay: int = 0):
        """Add a directed edge."""
        edge = InteractionEdge(source, target, weight, interaction_type, delay)
        self.edges.append(edge)
        if source in self.adjacency:
            self.adjacency[source].append(target)
        else:
            self.adjacency[source] = [target]

    def build_default_anokhin_graph(self, dim: int = 10) -> Dict[str, str]:
        """
        Build the default functional system graph based on Anokhin's architecture.

        Returns mapping of agent_id → agent_type.
        """
        # Core functional system agents
        agents = {
            "afferent_synth": "afferent_synthesis",
            "decision_dom": "decision_dominance",
            "aar_main": "aar",
            "efferent_motor": "efferent_action",
            "efferent_visceral": "efferent_action",
            "reverse_afferent": "reverse_afferentation",
            "mismatch_detect": "mismatch_detector",
            "aar_backup": "aar",  # Compensatory system
        }

        for aid, atype in agents.items():
            self.add_agent_node(aid, atype)

        # Environment nodes
        self.add_env_node(EnvironmentNode("env_stress", "stressor", 0.0, 0.3))
        self.add_env_node(EnvironmentNode("env_support", "support", 0.0, 0.8))
        self.add_env_node(EnvironmentNode("env_nutrition", "nutrition", 0.0, 0.9))
        self.add_env_node(EnvironmentNode("env_social", "social", 0.0, 0.5))
        self.add_env_node(EnvironmentNode("env_drones", "stressor", 0.0, 0.1))  # War stressor

        # Build Anokhin's circuit: Afferent Synthesis → Decision → AAR → Efferent → Feedback → Mismatch
        self.add_edge("afferent_synth", "decision_dom", 0.8, "excitatory")
        self.add_edge("decision_dom", "aar_main", 0.9, "prediction")
        self.add_edge("aar_main", "efferent_motor", 0.7, "prediction")
        self.add_edge("aar_main", "efferent_visceral", 0.6, "prediction")
        self.add_edge("efferent_motor", "reverse_afferent", 0.5, "feedback")
        self.add_edge("efferent_visceral", "reverse_afferent", 0.5, "feedback")
        self.add_edge("reverse_afferent", "mismatch_detect", 0.9, "feedback")
        self.add_edge("aar_main", "mismatch_detect", 0.9, "prediction")
        self.add_edge("mismatch_detect", "afferent_synth", 0.7, "inhibitory")
        self.add_edge("mismatch_detect", "decision_dom", 0.5, "inhibitory")

        # Compensatory system
        self.add_edge("decision_dom", "aar_backup", 0.4, "prediction")
        self.add_edge("aar_backup", "efferent_motor", 0.3, "prediction")

        # Environment → afferent synthesis
        self.add_edge("env_stress", "afferent_synth", -0.6, "inhibitory")
        self.add_edge("env_support", "afferent_synth", 0.5, "excitatory")
        self.add_edge("env_nutrition", "afferent_synth", 0.4, "excitatory")
        self.add_edge("env_social", "afferent_synth", 0.3, "excitatory")
        self.add_edge("env_drones", "afferent_synth", -0.8, "inhibitory")
        self.add_edge("env_drones", "mismatch_detect", 0.9, "excitatory")

        return agents

    def propagate(self, node_values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Propagate values through the graph (one step)."""
        new_values = {}
        for node_id in self.adjacency:
            if node_id in node_values:
                incoming = node_values[node_id].copy() * 0.1  # Self-connection
                # Collect from incoming edges
                for edge in self.edges:
                    if edge.target == node_id and edge.source in node_values:
                        contribution = edge.weight * node_values[edge.source]
                        if edge.interaction_type == "inhibitory":
                            contribution = -contribution
                        incoming += contribution
                new_values[node_id] = incoming
        return new_values

    def get_stress_paths(self) -> List[List[str]]:
        """Find all paths from stressors to mismatch detector."""
        # Simplified: return paths from env_drones and env_stress to mismatch_detect
        paths = []
        for stressor in ["env_drones", "env_stress"]:
            paths.append([stressor, "afferent_synth", "decision_dom",
                         "aar_main", "mismatch_detect"])
            paths.append([stressor, "mismatch_detect"])
        return paths

    def get_healing_paths(self) -> List[List[str]]:
        """Find paths that can reduce foam (healing circuits)."""
        return [
            ["env_support", "afferent_synth", "decision_dom", "aar_main"],
            ["env_nutrition", "afferent_synth", "decision_dom", "aar_main"],
            ["aar_backup", "efferent_motor", "reverse_afferent"],  # Compensatory
        ]