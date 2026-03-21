"""Module 1: Data Representation (Network Structure).

Defines the core data structures for representing network topologies,
including nodes with vulnerability indices and various graph generators.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class NodeState(Enum):
    """Possible states for a network node."""

    SAFE = "safe"
    INFECTED = "infected"


class Node:
    """A single node in the network.

    Attributes:
        node_id: Unique identifier for this node.
        state: Current infection state (SAFE or INFECTED).
        vulnerability: Float in [0, 1] representing how susceptible
            this node is to infection.  Higher means more vulnerable.
    """

    def __init__(
        self,
        node_id: int,
        vulnerability: float = 0.5,
        state: NodeState = NodeState.SAFE,
    ) -> None:
        if not 0.0 <= vulnerability <= 1.0:
            raise ValueError("vulnerability must be between 0.0 and 1.0")
        self.node_id = node_id
        self.vulnerability = vulnerability
        self.state = state

    def __repr__(self) -> str:
        return (
            f"Node(id={self.node_id}, state={self.state.value}, "
            f"vuln={self.vulnerability:.2f})"
        )


class Network:
    """An undirected network represented by an adjacency list.

    Attributes:
        nodes: Mapping from node id to ``Node`` instance.
        adjacency: Mapping from node id to the set of neighbor ids.
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.adjacency: Dict[int, Set[int]] = {}

    # ------------------------------------------------------------------
    # Basic graph operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node to the network."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists")
        self.nodes[node.node_id] = node
        self.adjacency[node.node_id] = set()

    def add_edge(self, id_a: int, id_b: int) -> None:
        """Add an undirected edge between two existing nodes."""
        if id_a not in self.nodes or id_b not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge")
        if id_a == id_b:
            raise ValueError("Self-loops are not allowed")
        self.adjacency[id_a].add(id_b)
        self.adjacency[id_b].add(id_a)

    def get_neighbors(self, node_id: int) -> List[Node]:
        """Return a list of ``Node`` objects adjacent to *node_id*."""
        return [self.nodes[nid] for nid in self.adjacency[node_id]]

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(nbrs) for nbrs in self.adjacency.values()) // 2

    def average_degree(self) -> float:
        """Return the average degree of the network."""
        if not self.nodes:
            return 0.0
        return 2.0 * self.edge_count() / self.node_count()

    def get_infected_nodes(self) -> List[Node]:
        """Return all nodes currently in the INFECTED state."""
        return [n for n in self.nodes.values() if n.state == NodeState.INFECTED]

    def get_safe_nodes(self) -> List[Node]:
        """Return all nodes currently in the SAFE state."""
        return [n for n in self.nodes.values() if n.state == NodeState.SAFE]

    def infection_ratio(self) -> float:
        """Return the fraction of nodes that are infected."""
        if not self.nodes:
            return 0.0
        return len(self.get_infected_nodes()) / self.node_count()

    def reset(self) -> None:
        """Reset all nodes to the SAFE state."""
        for node in self.nodes.values():
            node.state = NodeState.SAFE

    # ------------------------------------------------------------------
    # Graph generators (class methods)
    # ------------------------------------------------------------------

    @classmethod
    def _make_nodes(
        cls,
        n: int,
        vulnerability_range: Tuple[float, float] = (0.3, 0.9),
        rng: Optional[random.Random] = None,
    ) -> List[Node]:
        """Create *n* nodes with random vulnerabilities in *vulnerability_range*."""
        rng = rng or random.Random()
        lo, hi = vulnerability_range
        return [Node(i, vulnerability=rng.uniform(lo, hi)) for i in range(n)]

    @classmethod
    def random_graph(
        cls,
        n: int,
        average_degree: float,
        vulnerability_range: Tuple[float, float] = (0.3, 0.9),
        seed: Optional[int] = None,
    ) -> "Network":
        """Generate an Erdős–Rényi-style random graph.

        Each possible edge is included independently with probability
        ``p = average_degree / (n - 1)`` so that the expected average
        degree equals *average_degree*.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        rng = random.Random(seed)
        p = min(average_degree / (n - 1), 1.0)

        net = cls()
        for node in cls._make_nodes(n, vulnerability_range, rng):
            net.add_node(node)

        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    net.add_edge(i, j)
        return net

    @classmethod
    def star_graph(
        cls,
        n: int,
        vulnerability_range: Tuple[float, float] = (0.3, 0.9),
        seed: Optional[int] = None,
    ) -> "Network":
        """Generate a star graph with node 0 as the hub."""
        if n < 2:
            raise ValueError("n must be at least 2")
        rng = random.Random(seed)

        net = cls()
        for node in cls._make_nodes(n, vulnerability_range, rng):
            net.add_node(node)

        for i in range(1, n):
            net.add_edge(0, i)
        return net

    @classmethod
    def scale_free_graph(
        cls,
        n: int,
        m: int = 2,
        vulnerability_range: Tuple[float, float] = (0.3, 0.9),
        seed: Optional[int] = None,
    ) -> "Network":
        """Generate a scale-free network via the Barabási–Albert model.

        Starts with a fully connected core of *m* nodes and attaches
        each new node to *m* existing nodes chosen with probability
        proportional to their current degree (preferential attachment).
        """
        if m < 1 or m >= n:
            raise ValueError("m must be >= 1 and < n")
        rng = random.Random(seed)

        net = cls()
        nodes = cls._make_nodes(n, vulnerability_range, rng)
        for node in nodes:
            net.add_node(node)

        # Initial fully connected core of m nodes
        for i in range(m):
            for j in range(i + 1, m):
                net.add_edge(i, j)

        # Repeated degree list for preferential attachment sampling
        degree_list: List[int] = []
        for i in range(m):
            degree_list.extend([i] * (m - 1))

        for new_id in range(m, n):
            targets: Set[int] = set()
            while len(targets) < m:
                chosen = rng.choice(degree_list)
                if chosen != new_id:
                    targets.add(chosen)
            for t in targets:
                net.add_edge(new_id, t)
            for t in targets:
                degree_list.append(t)
                degree_list.append(new_id)

        return net
