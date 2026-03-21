"""Module 2: Core Algorithm (Random Walk Engine).

Implements the biased random walk where the probability of jumping to
a neighboring node is proportional to that neighbor's vulnerability index.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from network_attack.network import Network, Node, NodeState


class RandomWalkEngine:
    """Execute biased random walk steps on a network.

    The engine selects the next target among the **safe** neighbors of
    each infected node with probability proportional to their
    vulnerability index.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def get_safe_neighbors(self, network: Network, node_id: int) -> List[Node]:
        """Return safe (uninfected) neighbors of the given node."""
        return [
            n
            for n in network.get_neighbors(node_id)
            if n.state == NodeState.SAFE
        ]

    def build_probability_vector(self, neighbors: List[Node]) -> np.ndarray:
        """Build a probability distribution over *neighbors*.

        The probability of selecting a node is proportional to its
        vulnerability.  If all vulnerabilities are zero the distribution
        is uniform.
        """
        if not neighbors:
            return np.array([])
        weights = np.array([n.vulnerability for n in neighbors], dtype=np.float64)
        total = weights.sum()
        if total == 0:
            return np.ones(len(neighbors), dtype=np.float64) / len(neighbors)
        return weights / total

    def select_target(
        self, neighbors: List[Node], probabilities: np.ndarray
    ) -> Optional[Node]:
        """Randomly select a target node from *neighbors*.

        Returns ``None`` when there are no eligible neighbors.
        """
        if len(neighbors) == 0:
            return None
        idx = int(self._np_rng.choice(len(neighbors), p=probabilities))
        return neighbors[idx]

    def step_from_node(self, network: Network, node_id: int) -> Optional[Node]:
        """Perform one biased random walk step from *node_id*.

        Returns the newly infected ``Node`` or ``None`` if no safe
        neighbor was available.
        """
        safe_neighbors = self.get_safe_neighbors(network, node_id)
        if not safe_neighbors:
            return None
        probs = self.build_probability_vector(safe_neighbors)
        target = self.select_target(safe_neighbors, probs)
        if target is not None:
            target.state = NodeState.INFECTED
        return target

    def step_all_infected(self, network: Network) -> List[Node]:
        """Perform one step from **every** currently infected node.

        Returns the list of newly infected nodes in this step.
        """
        infected_ids = [n.node_id for n in network.get_infected_nodes()]
        newly_infected: List[Node] = []
        for nid in infected_ids:
            result = self.step_from_node(network, nid)
            if result is not None:
                newly_infected.append(result)
        return newly_infected
