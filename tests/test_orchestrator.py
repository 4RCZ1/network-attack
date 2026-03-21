"""Tests for Module 3: Simulation Orchestrator."""

import pytest

from network_attack.network import Network, Node, NodeState
from network_attack.orchestrator import SimulationOrchestrator
from network_attack.random_walk import RandomWalkEngine


def _line_network(n: int = 5) -> Network:
    """Build a simple line graph: 0-1-2-..-(n-1)."""
    net = Network()
    for i in range(n):
        net.add_node(Node(i, vulnerability=0.8))
    for i in range(n - 1):
        net.add_edge(i, i + 1)
    return net


class TestSimulationOrchestrator:
    def test_set_patient_zero(self):
        net = _line_network()
        orch = SimulationOrchestrator(net)
        orch.set_patient_zero([0])
        assert net.nodes[0].state == NodeState.INFECTED
        assert all(
            net.nodes[i].state == NodeState.SAFE for i in range(1, 5)
        )
        assert len(orch.history) == 1
        assert orch.history[0].infected_count == 1

    def test_patient_zero_invalid_id_raises(self):
        net = _line_network()
        orch = SimulationOrchestrator(net)
        with pytest.raises(ValueError):
            orch.set_patient_zero([99])

    def test_step_advances_epoch(self):
        net = _line_network()
        orch = SimulationOrchestrator(net, RandomWalkEngine(seed=0))
        orch.set_patient_zero([0])
        rec = orch.step()
        assert rec.epoch == 1

    def test_run_terminates(self):
        net = _line_network()
        orch = SimulationOrchestrator(net, RandomWalkEngine(seed=42))
        orch.set_patient_zero([0])
        history = orch.run(max_epochs=500)
        # The infection should eventually spread through the line
        assert history[-1].infection_ratio > 0.0
        assert len(history) >= 2

    def test_infection_curve_length_matches_history(self):
        net = _line_network()
        orch = SimulationOrchestrator(net, RandomWalkEngine(seed=0))
        orch.set_patient_zero([0])
        orch.run(max_epochs=50)
        curve = orch.infection_curve()
        assert len(curve) == len(orch.history)

    def test_epochs_to_full_infection(self):
        net = _line_network(3)
        orch = SimulationOrchestrator(net, RandomWalkEngine(seed=42))
        orch.set_patient_zero([1])  # middle node — fast spread
        orch.run(max_epochs=200)
        result = orch.epochs_to_full_infection()
        # Should eventually infect all 3 nodes
        assert result is not None
        assert result > 0

    def test_multiple_patient_zeros(self):
        net = _line_network(5)
        orch = SimulationOrchestrator(net)
        orch.set_patient_zero([0, 4])
        assert net.nodes[0].state == NodeState.INFECTED
        assert net.nodes[4].state == NodeState.INFECTED
        assert orch.history[0].infected_count == 2
