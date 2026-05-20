"""
Rejuvenation Scenario Simulation.

Models how GRA-Health-Swarm can guide an aged/diseased organism
toward rejuvenation through iterative AAR recalibration and
Φ_total minimization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import yaml
from core import RejuvenationLoop


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    sim_config = config['simulation']
    agent_config = config['agents']['initial_config']

    print("=" * 60)
    print("GRA-Health-Swarm: Rejuvenation Scenario")
    print("=" * 60)
    print(f"Description: {config['scenario']['description']}")
    print(f"Max steps: {sim_config['steps']}")
    print(f"Health threshold: {sim_config['health_threshold']}")
    print("-" * 60)

    # Initialize loop with aged parameters
    loop = RejuvenationLoop(dim=10, learning_rate=0.05)

    # Configure agents to simulate aged/diseased state
    loop.agents["afferent_synth"].adaptability = agent_config['afferent_synth']['adaptability']
    loop.agents["decision_dom"].adaptability = agent_config['decision_dom']['adaptability']
    loop.agents["aar_main"].adaptability = agent_config['aar_main']['recalibration_speed']
    loop.agents["reverse_afferent"].adaptability = agent_config['reverse_afferent']['feedback_accuracy']

    # Introduce initial "damage" — high Φ state
    damage_vector = np.random.normal(0, 0.5, loop.dim)
    for agent in loop.agents.values():
        agent.state = loop.target_state + damage_vector
        agent.prediction = loop.target_state.copy()  # Wrong prediction
        agent.compute_foam()

    print(f"Initial Φ_total: {loop.health_state.compute_total_foam({aid: a.foam for aid, a in loop.agents.items()}):.4f}")
    print("\nRunning rejuvenation protocol...")
    print(f"{'Step':<8} {'Φ_total':<12} {'Health':<10} {'AAR Acc':<10} {'Speed':<10} {'Status'}")
    print("-" * 60)

    rejuvenation_started = False
    rejuvenation_achieved = False

    for step in range(sim_config['steps']):
        # Apply interventions at specified frequencies
        ext_stimuli = loop.target_state + np.random.normal(0, 0.05, loop.dim)
        int_state = loop.target_state + np.random.normal(0, 0.08, loop.dim)
        
        stress = 0.05  # Low stress
        support = 0.9  # High support

        # Execute rejuvenation step
        metrics = loop.step(ext_stimuli, int_state, stress, support)

        # Apply specific rejuvenation interventions
        protocol = sim_config['rejuvenation_protocol']
        for intervention in protocol['interventions']:
            if step % intervention['frequency'] == 0:
                if intervention['type'] == "aar_reset":
                    loop.agents["aar_main"].adjust_prediction(intervention['intensity'])
                    loop.agents["aar_backup"].adjust_prediction(intervention['intensity'])
                elif intervention['type'] == "feedback_boost":
                    loop.agents["reverse_afferent"].adaptability = min(1.0,
                        loop.agents["reverse_afferent"].adaptability + intervention['intensity'])
                elif intervention['type'] == "adaptability_training":
                    for agent in loop.agents.values():
                        agent.adaptability = min(1.0, agent.adaptability + intervention['intensity'])

        # Track rejuvenation
        if loop.health_state.is_rejuvenating() and not rejuvenation_started:
            rejuvenation_started = True
            print(f"\n↗ Rejuvenation detected at step {step}!")

        # Print status
        if step % 100 == 0 or step == sim_config['steps'] - 1:
            summary = loop.get_current_state_summary()
            status = "HEALTHY" if summary['is_healthy'] else \
                    "REJUVENATING" if summary['is_rejuvenating'] else "AGED"
            
            print(f"{step:<8} {summary['phi_total']:<12.4f} "
                  f"{summary['health_index']:<10.4f} "
                  f"{summary['aar_accuracy']:<10.4f} "
                  f"{metrics.rejuvenation_speed:<10.6f} {status}")

        # Check if rejuvenation achieved
        if loop.health_state.is_healthy(threshold=sim_config['health_threshold']):
            rejuvenation_achieved = True
            print(f"\n✓ REJUVENATION ACHIEVED at step {step}!")
            print(f"  Φ_total = {loop.health_state.phi_total:.6f}")
            print(f"  Health Index = {loop.health_state.health_index:.4f}")
            break

    # Final report
    print("\n" + "=" * 60)
    print("REJUVENATION REPORT")
    print("=" * 60)
    summary = loop.get_current_state_summary()
    print(f"Achieved rejuvenation: {rejuvenation_achieved}")
    print(f"Final Φ_total: {summary['phi_total']:.6f}")
    print(f"Final Health Index: {summary['health_index']:.4f}")
    print(f"Rejuvenation Speed: {metrics.rejuvenation_speed:.6f}")
    print(f"Steps to health: {metrics.total_steps_to_health or 'Not reached'}")
    print(f"Relapses: {metrics.relapses}")

    if rejuvenation_achieved:
        print("\n" + "=" * 60)
        print("REJUVENATION MECHANISM SUMMARY")
        print("=" * 60)
        print("1. AAR Recalibration: Predictions adjusted to match reality")
        print("2. Feedback Strengthening: Reverse afferentation accuracy improved")
        print("3. Compensatory Systems: Backup AAR activated")
        print("4. Adaptability Training: All agents became more flexible")
        print("5. Φ_total → 0: System reached coherent healthy state")
        print("\nThis demonstrates Anokhin's principle:")
        print("The organism CAN restore function through dynamic")
        print("reorganization of functional systems — now formalized")
        print("as GRA nullification in a swarm of health agents.")


if __name__ == "__main__":
    main()