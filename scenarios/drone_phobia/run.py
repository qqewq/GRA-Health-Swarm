"""
Drone Phobia Scenario Simulation.

Models how chronic fear of drone attacks creates persistent Φ mismatch,
and how GRA-Health-Swarm's N_health operator can maintain coherence.

Key insight: GRA cannot stop the war, but it can help the system
survive without collapsing internally.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import yaml
from core import RejuvenationLoop, RejuvenationMetrics


def load_config():
    """Load scenario configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def drone_stress_generator(base_intensity: float, unpredictability: float):
    """Generate drone-related stress levels."""
    def generator(step: int) -> float:
        # Base stress + random attacks + periodicity
        base = base_intensity
        # Random attacks (unpredictable)
        if np.random.random() < unpredictability:
            attack_intensity = np.random.uniform(0.5, 1.0)
        else:
            attack_intensity = 0.0
        # Night/day cycle (more stress at night)
        time_of_day = (step % 24) / 24.0
        night_factor = 1.0 + 0.3 * np.sin(np.pi * time_of_day)
        # News cycle (periodic alarming reports)
        news_cycle = 0.2 * np.sin(2 * np.pi * step / 100)
        
        stress = (base + attack_intensity) * night_factor + news_cycle
        return np.clip(stress, 0.0, 1.0)
    return generator


def main():
    config = load_config()
    env = config['environment']
    agent_config = config['agents']['initial_config']
    sim_config = config['simulation']

    print("=" * 60)
    print("GRA-Health-Swarm: Drone Phobia Scenario")
    print("=" * 60)
    print(f"Description: {config['scenario']['description']}")
    print(f"Drone attack frequency: {env['drone_attacks']['frequency']}")
    print(f"Unpredictability: {env['drone_attacks']['unpredictability']}")
    print(f"Shelter availability: {env['shelters']['availability']}")
    print("-" * 60)

    # Initialize rejuvenation loop
    loop = RejuvenationLoop(dim=10, learning_rate=0.1)

    # Configure agents based on scenario
    loop.agents["afferent_synth"].adaptability = agent_config['afferent_synth']['adaptability']
    loop.agents["decision_dom"].adaptability = agent_config['decision_dom']['adaptability']
    loop.agents["decision_dom"].dominance = agent_config['decision_dom']['dominance']
    loop.agents["mismatch_detect"].stress_level = agent_config['mismatch_detect']['sensitivity']

    # Set up stress generator
    stress_gen = drone_stress_generator(
        env['drone_attacks']['intensity'],
        env['drone_attacks']['unpredictability']
    )

    # Also model social support
    social_support = env['social_support']['family_support']

    # Run simulation
    print("\nRunning simulation...")
    print(f"{'Step':<8} {'Φ_total':<12} {'Health':<10} {'Stress':<10} {'Status'}")
    print("-" * 60)

    for step in range(sim_config['steps']):
        # Generate inputs
        ext_stimuli = loop.target_state + np.random.normal(0, 0.1, loop.dim)
        int_state = loop.target_state + np.random.normal(0, 0.15, loop.dim)
        
        # Add drone stress
        drone_stress = stress_gen(step)
        support = social_support * (1.0 - 0.5 * drone_stress)  # Support decreases under stress

        # Execute step
        metrics = loop.step(ext_stimuli, int_state, drone_stress, support)

        # Print status periodically
        if step % 50 == 0 or step == sim_config['steps'] - 1:
            summary = loop.get_current_state_summary()
            status = "HEALTHY" if summary['is_healthy'] else \
                    "DISEASED" if summary['is_diseased'] else \
                    "REJUVENATING" if summary['is_rejuvenating'] else "STRESSED"
            
            print(f"{step:<8} {summary['phi_total']:<12.4f} "
                  f"{summary['health_index']:<10.4f} "
                  f"{summary['stress_level']:<10.4f} {status}")

        # Early stopping if healthy
        if loop.health_state.is_healthy(threshold=sim_config['health_threshold']):
            print(f"\n✓ System reached healthy state at step {step}")
            break

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    summary = loop.get_current_state_summary()
    print(f"Final Φ_total: {summary['phi_total']:.4f}")
    print(f"Final Health Index: {summary['health_index']:.4f}")
    print(f"Final Stress Level: {summary['stress_level']:.4f}")
    print(f"Health Trend: {summary['trend']}")
    print(f"Chronic Stress Risk: {summary['chronic_stress_risk']}")
    print(f"Rejuvenation Speed: {metrics.rejuvenation_speed:.6f}")
    print(f"Relapses: {metrics.relapses}")
    print(f"Total Nullification Steps: {len(loop.null_operator.history)}")

    if summary['is_healthy']:
        print("\n✓ The system maintains coherence despite drone threat.")
        print("  Key mechanism: AAR recalibration + adaptability increase.")
    elif summary['is_rejuvenating']:
        print("\n↗ System is actively rejuvenating (Φ decreasing).")
    else:
        print("\n⚠ System remains in chronic stress.")
        print("  Consider: increasing social support, improving information clarity,")
        print("  or adding shelter availability to reduce Φ.")

    # Show healing vs stress paths
    print("\n" + "-" * 60)
    print("Healing Paths (from Interaction Graph):")
    for path in loop.interaction_graph.get_healing_paths():
        print(f"  {' → '.join(path)}")
    print("\nStress Paths:")
    for path in loop.interaction_graph.get_stress_paths():
        print(f"  {' → '.join(path)}")


if __name__ == "__main__":
    main()