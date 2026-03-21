"""Module 2: Core Algorithm (Random Walk Engine).

Implements the biased random walk where the probability of jumping to
a neighboring node is proportional to that neighbor's vulnerability index.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from network_attack.network import Network, Node, NodeState


@dataclass
class StepDetail:
    """Detailed result of a single random walk step from an infected node.

    Attributes:
        source_id: The infected node that attempted to spread.
        considered_target_ids: IDs of safe neighbors that were candidates.
        chosen_target_id: ID of the node actually infected, or ``None``.
    """

    source_id: int
    considered_target_ids: List[int] = field(default_factory=list)
    chosen_target_id: Optional[int] = None


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

    # ------------------------------------------------------------------
    # Detailed variants that also track considered / chosen edges
    # ------------------------------------------------------------------

    def step_from_node_detailed(
        self, network: Network, node_id: int
    ) -> StepDetail:
        """Like :meth:`step_from_node` but return a :class:`StepDetail`.

        The random-number sequence is identical to ``step_from_node`` so
        simulations using detailed methods are fully reproducible.
        """
        safe_neighbors = self.get_safe_neighbors(network, node_id)
        if not safe_neighbors:
            return StepDetail(source_id=node_id)
        probs = self.build_probability_vector(safe_neighbors)
        target = self.select_target(safe_neighbors, probs)
        considered = [n.node_id for n in safe_neighbors]
        chosen = target.node_id if target is not None else None
        if target is not None:
            target.state = NodeState.INFECTED
        return StepDetail(
            source_id=node_id,
            considered_target_ids=considered,
            chosen_target_id=chosen,
        )

    def step_all_infected_detailed(
        self, network: Network
    ) -> Tuple[List[Node], List[StepDetail]]:
        """Like :meth:`step_all_infected` but also return per-source details.

        Returns ``(newly_infected, details)`` where *details* is a list
        of :class:`StepDetail` entries (one per currently infected node).
        """
        infected_ids = [n.node_id for n in network.get_infected_nodes()]
        newly_infected: List[Node] = []
        details: List[StepDetail] = []
        for nid in infected_ids:
            detail = self.step_from_node_detailed(network, nid)
            details.append(detail)
            if detail.chosen_target_id is not None:
                newly_infected.append(network.nodes[detail.chosen_target_id])
        return newly_infected, details
