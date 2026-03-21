"""Demo entry point for Attack Propagation in Complex Networks.

Run with:
    python main.py
"""

from network_attack.network import Network, NodeState
from network_attack.random_walk import RandomWalkEngine
from network_attack.orchestrator import SimulationOrchestrator
from network_attack.analytics import AnalyticsEngine
from network_attack.visualization import Visualization


def demo_single_run() -> None:
    """Run a single simulation and print epoch-by-epoch stats."""
    print("=== Single Simulation Run ===")
    net = Network.scale_free_graph(50, m=2, seed=7)
    print(f"Network: {net.node_count()} nodes, {net.edge_count()} edges, "
          f"avg degree {net.average_degree():.2f}")

    engine = RandomWalkEngine(seed=42)
    orch = SimulationOrchestrator(net, engine)
    orch.set_patient_zero([0])

    history = orch.run(max_epochs=200)
    for rec in history:
        bar = "#" * int(rec.infection_ratio * 40)
        print(f"  Epoch {rec.epoch:3d}: {rec.infection_ratio:5.1%} [{bar}]")

    epochs = orch.epochs_to_full_infection()
    if epochs is not None:
        print(f"Full infection reached at epoch {epochs}")
    else:
        print(f"Simulation ended at {history[-1].infection_ratio:.1%} infection")
    print()


def demo_batch_analysis() -> None:
    """Run batch simulations across different average degrees."""
    print("=== Batch Analysis: Average Degree vs Infection Speed ===")
    analytics = AnalyticsEngine(seed=0)

    degree_results = []
    for avg_deg in [2, 4, 6, 8]:
        factory = lambda d=avg_deg: Network.random_graph(
            50, average_degree=d, seed=None
        )
        result = analytics.run_batch(
            factory, patient_zero_ids=[0], num_runs=20, max_epochs=300
        )
        print(f"  avg_degree={avg_deg}: mean epochs to full infection = "
              f"{result.mean_epochs_to_full_infection}")
        degree_results.append((float(avg_deg), result))

    r, p = AnalyticsEngine.degree_vs_infection_speed(degree_results)
    if r is not None:
        print(f"  Pearson r = {r:.3f}, p ≈ {p:.4f}")
    print()


def demo_defense_strategy() -> None:
    """Show how high-resistance nodes slow infection."""
    print("=== Defense Strategy: High-Resistance Nodes ===")

    # Baseline (no defense)
    net_base = Network.random_graph(40, average_degree=4.0, seed=10)
    orch_base = SimulationOrchestrator(net_base, RandomWalkEngine(seed=0))
    orch_base.set_patient_zero([0])
    orch_base.run(max_epochs=300)

    # With defense: set some nodes to very low vulnerability
    net_def = Network.random_graph(40, average_degree=4.0, seed=10)
    for nid in [5, 10, 15, 20, 25]:
        if nid in net_def.nodes:
            net_def.nodes[nid].vulnerability = 0.01  # "firewall" nodes
    orch_def = SimulationOrchestrator(net_def, RandomWalkEngine(seed=0))
    orch_def.set_patient_zero([0])
    orch_def.run(max_epochs=300)

    base_epochs = orch_base.epochs_to_full_infection()
    def_epochs = orch_def.epochs_to_full_infection()
    print(f"  No defense:   full infection at epoch {base_epochs}")
    print(f"  With defense: full infection at epoch {def_epochs}")
    print()


if __name__ == "__main__":
    demo_single_run()
    demo_batch_analysis()
    demo_defense_strategy()