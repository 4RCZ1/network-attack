"""Module 5: Presentation Layer (Visualization).

Provides functions for rendering the network graph, infection-growth
curves, and optional step-by-step interactive controls.  Uses
*matplotlib* exclusively so the project stays dependency-light.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from network_attack.network import Network, NodeState
from network_attack.orchestrator import SimulationOrchestrator


def _spring_layout(
    network: Network, iterations: int = 50, seed: int = 42
) -> Dict[int, Tuple[float, float]]:
    """Compute a simple force-directed (spring) layout for *network*.

    Returns a mapping from node id to ``(x, y)`` position.
    """
    rng = np.random.default_rng(seed)
    pos = {nid: rng.uniform(-1, 1, size=2) for nid in network.nodes}

    k = 1.0 / (np.sqrt(network.node_count()) + 1e-6)

    for _ in range(iterations):
        displacement: Dict[int, np.ndarray] = {
            nid: np.zeros(2) for nid in network.nodes
        }

        nodes_list = list(network.nodes.keys())
        for i, u in enumerate(nodes_list):
            for v in nodes_list[i + 1 :]:
                delta = pos[u] - pos[v]
                dist = np.linalg.norm(delta) + 1e-9
                repulsion = (k**2 / dist) * (delta / dist)
                displacement[u] += repulsion
                displacement[v] -= repulsion

        for u in network.nodes:
            for v in network.adjacency[u]:
                if v > u:
                    delta = pos[u] - pos[v]
                    dist = np.linalg.norm(delta) + 1e-9
                    attraction = (dist / k) * (delta / dist)
                    displacement[u] -= attraction
                    displacement[v] += attraction

        for nid in network.nodes:
            norm = np.linalg.norm(displacement[nid]) + 1e-9
            pos[nid] += 0.1 * displacement[nid] / norm

    return {nid: (float(p[0]), float(p[1])) for nid, p in pos.items()}


class Visualization:
    """High-level helpers for plotting network simulations."""

    COLOR_SAFE = "#3498db"
    COLOR_INFECTED = "#e74c3c"

    # ------------------------------------------------------------------
    # Graph rendering
    # ------------------------------------------------------------------

    @staticmethod
    def draw_network(
        network: Network,
        pos: Optional[Dict[int, Tuple[float, float]]] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Network",
    ) -> plt.Axes:
        """Draw the network with node colors reflecting their state."""
        if pos is None:
            pos = _spring_layout(network)
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        # Draw edges
        for u in network.nodes:
            for v in network.adjacency[u]:
                if v > u:
                    xu, yu = pos[u]
                    xv, yv = pos[v]
                    ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=0.5, zorder=1)

        # Draw nodes
        for nid, node in network.nodes.items():
            x, y = pos[nid]
            color = (
                Visualization.COLOR_INFECTED
                if node.state == NodeState.INFECTED
                else Visualization.COLOR_SAFE
            )
            size = 40 + 160 * node.vulnerability
            ax.scatter(x, y, s=size, c=color, edgecolors="black", linewidths=0.5, zorder=2)

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
        return ax

    # ------------------------------------------------------------------
    # Infection growth curve
    # ------------------------------------------------------------------

    @staticmethod
    def plot_infection_curve(
        curves: Dict[str, List[float]],
        ax: Optional[plt.Axes] = None,
        title: str = "Infection Growth",
    ) -> plt.Axes:
        """Plot one or more infection-ratio curves.

        *curves* maps a label string to a list of infection ratios
        (one per epoch).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for label, data in curves.items():
            ax.plot(range(len(data)), data, label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Infection Ratio")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Step-by-step interactive viewer
    # ------------------------------------------------------------------

    @staticmethod
    def interactive_step(
        orchestrator: SimulationOrchestrator,
        pos: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> None:
        """Open a matplotlib window with *Next Step* / *Reset* buttons.

        Each click on *Next Step* advances the simulation by one epoch
        and redraws the graph.
        """
        if pos is None:
            pos = _spring_layout(orchestrator.network)

        fig, (ax_graph, ax_curve) = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(bottom=0.2)

        ratios: List[float] = [orchestrator.network.infection_ratio()]

        def _redraw() -> None:
            ax_graph.clear()
            Visualization.draw_network(
                orchestrator.network,
                pos=pos,
                ax=ax_graph,
                title=f"Epoch {orchestrator.current_epoch}",
            )
            ax_curve.clear()
            ax_curve.plot(range(len(ratios)), ratios, color=Visualization.COLOR_INFECTED)
            ax_curve.set_xlabel("Epoch")
            ax_curve.set_ylabel("Infection Ratio")
            ax_curve.set_ylim(-0.05, 1.05)
            ax_curve.set_title("Infection Growth")
            ax_curve.grid(True, alpha=0.3)
            fig.canvas.draw_idle()

        def _on_next(event: object) -> None:
            if orchestrator.network.infection_ratio() < 1.0:
                orchestrator.step()
                ratios.append(orchestrator.network.infection_ratio())
                _redraw()

        def _on_reset(event: object) -> None:
            initial_ids = (
                orchestrator.history[0].newly_infected_ids
                if orchestrator.history
                else [0]
            )
            orchestrator.set_patient_zero(initial_ids)
            ratios.clear()
            ratios.append(orchestrator.network.infection_ratio())
            _redraw()

        ax_next = plt.axes((0.35, 0.05, 0.12, 0.06))
        ax_reset = plt.axes((0.53, 0.05, 0.12, 0.06))
        btn_next = Button(ax_next, "Next Step")
        btn_reset = Button(ax_reset, "Reset")
        btn_next.on_clicked(_on_next)
        btn_reset.on_clicked(_on_reset)

        _redraw()
        plt.show()
