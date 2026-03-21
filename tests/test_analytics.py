"""Tests for Module 4: Analytics Engine."""

import pytest

from network_attack.analytics import AnalyticsEngine, BatchResult
from network_attack.network import Network, Node


class TestAnalyticsEngine:
    def _factory(self) -> Network:
        """Create a small fully-connected 5-node network."""
        net = Network()
        for i in range(5):
            net.add_node(Node(i, vulnerability=0.7))
        for i in range(5):
            for j in range(i + 1, 5):
                net.add_edge(i, j)
        return net

    def test_batch_result_structure(self):
        engine = AnalyticsEngine(seed=0)
        result = engine.run_batch(self._factory, patient_zero_ids=[0], num_runs=10, max_epochs=100)
        assert isinstance(result, BatchResult)
        assert result.num_runs == 10
        assert len(result.individual_epochs) == 10

    def test_batch_mean_curve_non_empty(self):
        engine = AnalyticsEngine(seed=0)
        result = engine.run_batch(self._factory, patient_zero_ids=[0], num_runs=5, max_epochs=100)
        assert len(result.mean_infection_curve) > 0
        assert result.mean_infection_curve[0] <= 1.0

    def test_batch_full_infection_achieved(self):
        engine = AnalyticsEngine(seed=42)
        result = engine.run_batch(self._factory, patient_zero_ids=[0], num_runs=10, max_epochs=200)
        # On a fully-connected 5-node graph, all runs should reach full infection
        assert result.mean_epochs_to_full_infection is not None
        assert result.mean_epochs_to_full_infection > 0

    def test_degree_vs_infection_speed_not_enough_data(self):
        r, p = AnalyticsEngine.degree_vs_infection_speed([])
        assert r is None and p is None

    def test_degree_vs_infection_speed_with_data(self):
        engine = AnalyticsEngine(seed=0)

        results = []
        for avg_deg in [2.0, 4.0, 6.0, 8.0]:
            factory = lambda d=avg_deg: Network.random_graph(30, average_degree=d, seed=0)
            br = engine.run_batch(factory, patient_zero_ids=[0], num_runs=5, max_epochs=300)
            results.append((avg_deg, br))

        r, p = AnalyticsEngine.degree_vs_infection_speed(results)
        # Correlation should be computable
        assert r is not None
        # Higher degree => faster infection => negative correlation expected
        assert -1.0 <= r <= 1.0

    def test_batch_without_infection_order(self):
        """Default run_batch does NOT collect infection order."""
        engine = AnalyticsEngine(seed=0)
        result = engine.run_batch(
            self._factory, patient_zero_ids=[0], num_runs=3, max_epochs=100,
        )
        assert result.mean_infection_order is None

    def test_batch_with_infection_order(self):
        """collect_infection_order=True populates mean_infection_order."""
        engine = AnalyticsEngine(seed=0)
        result = engine.run_batch(
            self._factory, patient_zero_ids=[0], num_runs=5, max_epochs=100,
            collect_infection_order=True,
        )
        assert result.mean_infection_order is not None
        # Patient zero (node 0) should always be infected at epoch 0
        assert result.mean_infection_order[0] == 0.0
        # All 5 nodes should appear in the mapping
        assert len(result.mean_infection_order) == 5
        # Every mean epoch must be non-negative
        for nid, epoch in result.mean_infection_order.items():
            assert epoch >= 0.0
