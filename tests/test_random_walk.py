"""Tests for Module 2: Core Algorithm (Random Walk Engine)."""

import numpy as np
import pytest

from network_attack.network import Network, Node, NodeState
from network_attack.random_walk import RandomWalkEngine, StepDetail


def _triangle_network() -> Network:
    """Build a simple 3-node triangle for testing."""
    net = Network()
    net.add_node(Node(0, vulnerability=0.9))
    net.add_node(Node(1, vulnerability=0.1))
    net.add_node(Node(2, vulnerability=0.5))
    net.add_edge(0, 1)
    net.add_edge(0, 2)
    net.add_edge(1, 2)
    return net


class TestRandomWalkEngine:
    def test_get_safe_neighbors_excludes_infected(self):
        net = _triangle_network()
        net.nodes[0].state = NodeState.INFECTED
        net.nodes[1].state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=0)
        safe = engine.get_safe_neighbors(net, 0)
        assert len(safe) == 1
        assert safe[0].node_id == 2

    def test_probability_vector_sums_to_one(self):
        net = _triangle_network()
        engine = RandomWalkEngine()
        neighbors = net.get_neighbors(0)
        probs = engine.build_probability_vector(neighbors)
        assert probs.sum() == pytest.approx(1.0)

    def test_probability_vector_empty(self):
        engine = RandomWalkEngine()
        probs = engine.build_probability_vector([])
        assert len(probs) == 0

    def test_probability_vector_proportional(self):
        net = _triangle_network()
        engine = RandomWalkEngine()
        # Neighbors of 0 are node 1 (vuln 0.1) and node 2 (vuln 0.5)
        neighbors = engine.get_safe_neighbors(net, 0)
        neighbors.sort(key=lambda n: n.node_id)
        probs = engine.build_probability_vector(neighbors)
        # node1: 0.1/(0.1+0.5) ≈ 0.167, node2: 0.5/0.6 ≈ 0.833
        assert probs[0] == pytest.approx(0.1 / 0.6)
        assert probs[1] == pytest.approx(0.5 / 0.6)

    def test_select_target_returns_none_when_empty(self):
        engine = RandomWalkEngine(seed=0)
        result = engine.select_target([], np.array([]))
        assert result is None

    def test_step_from_node_infects_target(self):
        net = _triangle_network()
        net.nodes[0].state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=42)
        result = engine.step_from_node(net, 0)
        assert result is not None
        assert result.state == NodeState.INFECTED

    def test_step_from_node_no_safe_neighbors(self):
        net = _triangle_network()
        for n in net.nodes.values():
            n.state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=0)
        result = engine.step_from_node(net, 0)
        assert result is None

    def test_step_all_infected(self):
        net = _triangle_network()
        net.nodes[0].state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=42)
        newly = engine.step_all_infected(net)
        assert len(newly) >= 1
        assert all(n.state == NodeState.INFECTED for n in newly)

    # ------------------------------------------------------------------
    # Detailed step methods
    # ------------------------------------------------------------------

    def test_step_from_node_detailed_returns_step_detail(self):
        net = _triangle_network()
        net.nodes[0].state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=42)
        detail = engine.step_from_node_detailed(net, 0)
        assert isinstance(detail, StepDetail)
        assert detail.source_id == 0
        # Node 0 has two safe neighbors (1, 2)
        assert set(detail.considered_target_ids) == {1, 2}
        # Exactly one is chosen
        assert detail.chosen_target_id in {1, 2}
        # The chosen node should now be infected
        assert net.nodes[detail.chosen_target_id].state == NodeState.INFECTED

    def test_step_from_node_detailed_no_safe_neighbors(self):
        net = _triangle_network()
        for n in net.nodes.values():
            n.state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=0)
        detail = engine.step_from_node_detailed(net, 0)
        assert detail.source_id == 0
        assert detail.considered_target_ids == []
        assert detail.chosen_target_id is None

    def test_step_all_infected_detailed_structure(self):
        net = _triangle_network()
        net.nodes[0].state = NodeState.INFECTED
        engine = RandomWalkEngine(seed=42)
        newly, details = engine.step_all_infected_detailed(net)
        # One infected source ⇒ one detail entry
        assert len(details) == 1
        assert details[0].source_id == 0
        assert len(newly) >= 1
        for n in newly:
            assert n.state == NodeState.INFECTED

    def test_detailed_and_normal_produce_same_result(self):
        """Verify detailed methods produce identical outcomes to normal."""
        # Normal run
        net1 = _triangle_network()
        net1.nodes[0].state = NodeState.INFECTED
        engine1 = RandomWalkEngine(seed=99)
        newly1 = engine1.step_all_infected(net1)

        # Detailed run with same seed
        net2 = _triangle_network()
        net2.nodes[0].state = NodeState.INFECTED
        engine2 = RandomWalkEngine(seed=99)
        newly2, _ = engine2.step_all_infected_detailed(net2)

        ids1 = sorted(n.node_id for n in newly1)
        ids2 = sorted(n.node_id for n in newly2)
        assert ids1 == ids2
