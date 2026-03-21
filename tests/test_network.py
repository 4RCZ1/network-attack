"""Tests for Module 1: Data Representation (Network Structure)."""

import pytest

from network_attack.network import Network, Node, NodeState


class TestNode:
    def test_default_state_is_safe(self):
        node = Node(0)
        assert node.state == NodeState.SAFE

    def test_custom_vulnerability(self):
        node = Node(1, vulnerability=0.8)
        assert node.vulnerability == pytest.approx(0.8)

    def test_vulnerability_out_of_range(self):
        with pytest.raises(ValueError):
            Node(0, vulnerability=1.5)
        with pytest.raises(ValueError):
            Node(0, vulnerability=-0.1)

    def test_repr(self):
        node = Node(3, vulnerability=0.75, state=NodeState.INFECTED)
        assert "id=3" in repr(node)
        assert "infected" in repr(node)


class TestNetwork:
    def test_add_node(self):
        net = Network()
        net.add_node(Node(0))
        assert net.node_count() == 1

    def test_add_duplicate_node_raises(self):
        net = Network()
        net.add_node(Node(0))
        with pytest.raises(ValueError):
            net.add_node(Node(0))

    def test_add_edge(self):
        net = Network()
        net.add_node(Node(0))
        net.add_node(Node(1))
        net.add_edge(0, 1)
        assert net.edge_count() == 1
        assert 1 in net.adjacency[0]
        assert 0 in net.adjacency[1]

    def test_self_loop_raises(self):
        net = Network()
        net.add_node(Node(0))
        with pytest.raises(ValueError):
            net.add_edge(0, 0)

    def test_edge_missing_node_raises(self):
        net = Network()
        net.add_node(Node(0))
        with pytest.raises(ValueError):
            net.add_edge(0, 99)

    def test_get_neighbors(self):
        net = Network()
        for i in range(3):
            net.add_node(Node(i))
        net.add_edge(0, 1)
        net.add_edge(0, 2)
        neighbors = net.get_neighbors(0)
        assert len(neighbors) == 2

    def test_average_degree(self):
        net = Network()
        for i in range(4):
            net.add_node(Node(i))
        # 0-1, 0-2, 0-3  => degrees: 3, 1, 1, 1 => avg = 6/4 = 1.5
        net.add_edge(0, 1)
        net.add_edge(0, 2)
        net.add_edge(0, 3)
        assert net.average_degree() == pytest.approx(1.5)

    def test_infection_ratio(self):
        net = Network()
        for i in range(4):
            net.add_node(Node(i))
        net.nodes[0].state = NodeState.INFECTED
        assert net.infection_ratio() == pytest.approx(0.25)

    def test_reset(self):
        net = Network()
        for i in range(3):
            net.add_node(Node(i))
        net.nodes[0].state = NodeState.INFECTED
        net.nodes[1].state = NodeState.INFECTED
        net.reset()
        assert all(n.state == NodeState.SAFE for n in net.nodes.values())


class TestGraphGenerators:
    def test_random_graph_node_count(self):
        net = Network.random_graph(20, average_degree=4.0, seed=42)
        assert net.node_count() == 20

    def test_random_graph_approximate_degree(self):
        net = Network.random_graph(200, average_degree=6.0, seed=42)
        assert abs(net.average_degree() - 6.0) < 2.0

    def test_star_graph_structure(self):
        net = Network.star_graph(10, seed=0)
        assert net.node_count() == 10
        assert net.edge_count() == 9
        assert len(net.adjacency[0]) == 9

    def test_scale_free_graph_node_count(self):
        net = Network.scale_free_graph(50, m=2, seed=7)
        assert net.node_count() == 50

    def test_random_graph_too_small_raises(self):
        with pytest.raises(ValueError):
            Network.random_graph(1, average_degree=1.0)

    def test_scale_free_invalid_m_raises(self):
        with pytest.raises(ValueError):
            Network.scale_free_graph(5, m=5)

    def test_random_graph_remove_orphans(self):
        # Use a very low average degree to make orphans likely
        net = Network.random_graph(
            50, average_degree=0.5, seed=99, remove_orphans=True
        )
        for nid in net.nodes:
            assert len(net.adjacency[nid]) >= 1, f"Node {nid} is still an orphan"

    def test_random_graph_remove_orphans_all_isolated(self):
        # average_degree=0 means p=0, so all nodes are orphans
        net = Network.random_graph(
            10, average_degree=0.0, seed=0, remove_orphans=True
        )
        for nid in net.nodes:
            assert len(net.adjacency[nid]) >= 1, f"Node {nid} is still an orphan"

    def test_random_graph_remove_orphans_false_preserves_behavior(self):
        net_a = Network.random_graph(20, average_degree=4.0, seed=42)
        net_b = Network.random_graph(
            20, average_degree=4.0, seed=42, remove_orphans=False
        )
        assert net_a.edge_count() == net_b.edge_count()
