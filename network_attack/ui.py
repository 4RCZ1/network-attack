"""Streamlit UI for managing network attack simulations.

Launch with:
    streamlit run network_attack/ui.py
"""

from __future__ import annotations

import io
import sys
import os
import time

# Ensure the project root is on sys.path so 'network_attack' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from network_attack.analytics import AnalyticsEngine, BatchResult
from network_attack.network import Network, NodeState
from network_attack.orchestrator import EpochRecord, SimulationOrchestrator
from network_attack.random_walk import RandomWalkEngine

matplotlib.use("Agg")

# ------------------------------------------------------------------
# Layout helpers
# ------------------------------------------------------------------

COLOR_SAFE = "#3498db"
COLOR_INFECTED = "#000000"
COLOR_NEWLY_INFECTED = "#f39c12"
COLOR_EDGE_DEFAULT = "rgba(200,200,200,0.4)"
COLOR_EDGE_CONSIDERED = "#FFA500"
COLOR_EDGE_CHOSEN = "#DC143C"


def _vulnerability_color(vuln: float) -> str:
    """Return a hex color on a green→yellow→red gradient for *vuln* ∈ [0, 1].

    0.0 → pure green, 0.5 → yellow, 1.0 → pure red.
    """
    # Clamp to [0, 1]
    v = max(0.0, min(1.0, vuln))
    # Green-to-red via linear interpolation through yellow
    r = int(min(255, 2 * v * 255))
    g = int(min(255, 2 * (1 - v) * 255))
    return f"#{r:02x}{g:02x}00"


def _spring_layout(
    network: Network, iterations: int = 50, seed: int = 42
) -> Dict[int, Tuple[float, float]]:
    """Compute a force-directed layout identical to visualization.py."""
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
                dist = float(np.linalg.norm(delta)) + 1e-9
                repulsion = (k**2 / dist) * (delta / dist)
                displacement[u] += repulsion
                displacement[v] -= repulsion
        for u in network.nodes:
            for v in network.adjacency[u]:
                if v > u:
                    delta = pos[u] - pos[v]
                    dist = float(np.linalg.norm(delta)) + 1e-9
                    attraction = (dist / k) * (delta / dist)
                    displacement[u] -= attraction
                    displacement[v] += attraction
        for nid in network.nodes:
            norm = float(np.linalg.norm(displacement[nid])) + 1e-9
            pos[nid] += 0.1 * displacement[nid] / norm

    return {nid: (float(p[0]), float(p[1])) for nid, p in pos.items()}


def _draw_network_fig(
    network: Network,
    pos: Dict[int, Tuple[float, float]],
    title: str = "Network",
) -> plt.Figure:
    """Return a matplotlib Figure of the network."""
    fig, ax = plt.subplots(figsize=(7, 7))
    for u in network.nodes:
        for v in network.adjacency[u]:
            if v > u:
                xu, yu = pos[u]
                xv, yv = pos[v]
                ax.plot(
                    [xu, xv], [yu, yv], color="#cccccc", linewidth=0.5, zorder=1
                )
    for nid, node in network.nodes.items():
        x, y = pos[nid]
        if node.state == NodeState.INFECTED:
            color = COLOR_INFECTED
        else:
            color = _vulnerability_color(node.vulnerability)
        size = 40 + 160 * node.vulnerability
        ax.scatter(
            x, y, s=size, c=color, edgecolors="black", linewidths=0.5, zorder=2
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _draw_network_plotly(
    network: Network,
    pos: Dict[int, Tuple[float, float]],
    title: str = "Network",
    considered_edges: Optional[List[Tuple[int, int]]] = None,
    chosen_edges: Optional[List[Tuple[int, int]]] = None,
    newly_infected_ids: Optional[List[int]] = None,
) -> go.Figure:
    """Return a Plotly Figure of the network with edge highlighting.

    Parameters
    ----------
    network:
        The network to draw.
    pos:
        Mapping from node id to ``(x, y)`` position.
    title:
        Figure title.
    considered_edges:
        Edges the random walk evaluated (drawn in orange).
    chosen_edges:
        Edges the random walk actually traversed (drawn in crimson).
    newly_infected_ids:
        Node IDs infected this epoch (drawn in a distinct color).
    """
    considered_set: set[Tuple[int, int]] = set()
    chosen_set: set[Tuple[int, int]] = set()
    if considered_edges:
        for u, v in considered_edges:
            considered_set.add((min(u, v), max(u, v)))
    if chosen_edges:
        for u, v in chosen_edges:
            chosen_set.add((min(u, v), max(u, v)))
    newly_set = set(newly_infected_ids or [])

    # ---- Edge traces ----
    def _edge_coords(
        edge_list: set[Tuple[int, int]],
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        for u, v in edge_list:
            xu, yu = pos[u]
            xv, yv = pos[v]
            xs.extend([xu, xv, None])
            ys.extend([yu, yv, None])
        return xs, ys

    # Collect default edges (those not in considered or chosen)
    default_edges: set[Tuple[int, int]] = set()
    for u in network.nodes:
        for v in network.adjacency[u]:
            if v > u:
                key = (u, v)
                if key not in considered_set and key not in chosen_set:
                    default_edges.add(key)

    fig = go.Figure()

    # Default edges
    dx, dy = _edge_coords(default_edges)
    if dx:
        fig.add_trace(go.Scatter(
            x=dx, y=dy, mode="lines",
            line=dict(color=COLOR_EDGE_DEFAULT, width=1),
            hoverinfo="none", showlegend=False,
        ))

    # Considered edges (exclude those that are also chosen)
    considered_only = considered_set - chosen_set
    cx, cy = _edge_coords(considered_only)
    if cx:
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color=COLOR_EDGE_CONSIDERED, width=2.5),
            hoverinfo="none", name="Considered",
        ))

    # Chosen edges
    chx, chy = _edge_coords(chosen_set)
    if chx:
        fig.add_trace(go.Scatter(
            x=chx, y=chy, mode="lines",
            line=dict(color=COLOR_EDGE_CHOSEN, width=4),
            hoverinfo="none", name="Chosen",
        ))

    # ---- Node traces ----
    def _node_trace(
        node_ids: List[int], color: str, name: str, symbol: str = "circle",
    ) -> None:
        if not node_ids:
            return
        xs = [pos[nid][0] for nid in node_ids]
        ys = [pos[nid][1] for nid in node_ids]
        sizes = [8 + 16 * network.nodes[nid].vulnerability for nid in node_ids]
        texts = [
            f"Node {nid}<br>State: {network.nodes[nid].state.value}"
            f"<br>Vulnerability: {network.nodes[nid].vulnerability:.2f}"
            for nid in node_ids
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(
                size=sizes, color=color, symbol=symbol,
                line=dict(width=1, color="black"),
            ),
            text=texts, hoverinfo="text",
        ))

    def _node_trace_gradient(
        node_ids: List[int], name: str,
    ) -> None:
        """Draw safe nodes with per-node vulnerability gradient colors."""
        if not node_ids:
            return
        xs = [pos[nid][0] for nid in node_ids]
        ys = [pos[nid][1] for nid in node_ids]
        sizes = [8 + 16 * network.nodes[nid].vulnerability for nid in node_ids]
        colors = [_vulnerability_color(network.nodes[nid].vulnerability)
                  for nid in node_ids]
        texts = [
            f"Node {nid}<br>State: {network.nodes[nid].state.value}"
            f"<br>Vulnerability: {network.nodes[nid].vulnerability:.2f}"
            for nid in node_ids
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name=name,
            marker=dict(
                size=sizes, color=colors, symbol="circle",
                line=dict(width=1, color="black"),
            ),
            text=texts, hoverinfo="text",
        ))

    safe_ids = [
        nid for nid, node in network.nodes.items()
        if node.state == NodeState.SAFE
    ]
    prev_infected_ids = [
        nid for nid, node in network.nodes.items()
        if node.state == NodeState.INFECTED and nid not in newly_set
    ]
    newly_ids_present = [
        nid for nid in newly_set if nid in network.nodes
    ]

    _node_trace_gradient(safe_ids, "Safe (by vulnerability)")
    _node_trace(prev_infected_ids, COLOR_INFECTED, "Infected")
    _node_trace(newly_ids_present, COLOR_NEWLY_INFECTED, "Newly infected",
                symbol="diamond")

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            visible=False, scaleanchor="x",
        ),
        plot_bgcolor="white",
        width=700, height=700,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _draw_infection_heatmap(
    network: Network,
    pos: Dict[int, Tuple[float, float]],
    mean_infection_order: Dict[int, float],
    title: str = "Infection Heatmap",
) -> go.Figure:
    """Return a Plotly Figure showing mean infection order as a heatmap.

    Nodes are colored by the average epoch at which they were first
    infected across batch runs: dark purple (early) → bright yellow (late).
    """
    fig = go.Figure()

    # Draw edges
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for u in network.nodes:
        for v in network.adjacency[u]:
            if v > u:
                xu, yu = pos[u]
                xv, yv = pos[v]
                edge_x.extend([xu, xv, None])
                edge_y.extend([yu, yv, None])
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=COLOR_EDGE_DEFAULT, width=1),
            hoverinfo="none", showlegend=False,
        ))

    # Draw nodes with heatmap coloring
    node_ids = list(network.nodes.keys())
    xs = [pos[nid][0] for nid in node_ids]
    ys = [pos[nid][1] for nid in node_ids]
    heat_values = [mean_infection_order.get(nid, float("nan")) for nid in node_ids]
    sizes = [10 + 14 * network.nodes[nid].vulnerability for nid in node_ids]
    texts = [
        f"Node {nid}<br>Vulnerability: {network.nodes[nid].vulnerability:.2f}"
        f"<br>Mean infection epoch: {mean_infection_order.get(nid, float('nan')):.1f}"
        for nid in node_ids
    ]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="Infection order",
        marker=dict(
            size=sizes,
            color=heat_values,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Mean epoch"),
            line=dict(width=1, color="black"),
        ),
        text=texts, hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            visible=False, scaleanchor="x",
        ),
        plot_bgcolor="white",
        width=700, height=700,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _draw_curve_fig(
    curves: Dict[str, List[float]],
    title: str = "Infection Growth",
    highlight_epoch: Optional[int] = None,
    full_curve: Optional[List[float]] = None,
) -> plt.Figure:
    """Return a matplotlib Figure with infection curves.

    Parameters
    ----------
    curves:
        Mapping of label → list of infection ratios to plot.
    title:
        Figure title.
    highlight_epoch:
        If given, draw a marker dot at this epoch on the first curve.
    full_curve:
        If given, draw the full trajectory as a faint ghost line behind
        the main curves so the viewer can anticipate the overall shape.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Ghost preview of the complete curve (shown during replay)
    if full_curve is not None:
        ax.plot(
            range(len(full_curve)),
            full_curve,
            color="#e74c3c",
            alpha=0.15,
            linewidth=1.5,
            linestyle="--",
            label="_ghost",
        )
        ax.set_xlim(-0.5, len(full_curve) - 0.5)

    use_single_color = len(curves) == 1

    for label, data in curves.items():
        epochs = list(range(len(data)))
        plot_kwargs: Dict[str, Any] = {"label": label, "linewidth": 2}
        if use_single_color:
            plot_kwargs["color"] = "#e74c3c"
        (line,) = ax.plot(epochs, data, **plot_kwargs)
        # Filled area under the curve
        ax.fill_between(
            epochs,
            data,
            alpha=0.18,
            color=line.get_color(),
        )

        # Animated marker at the current epoch
        if highlight_epoch is not None and 0 <= highlight_epoch < len(data):
            ax.plot(
                highlight_epoch,
                data[highlight_epoch],
                "o",
                color=line.get_color(),
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=5,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Infection Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Network snapshot helpers for replay
# ------------------------------------------------------------------


def _snapshot_states(network: Network) -> Dict[int, NodeState]:
    """Capture current infection states of all nodes."""
    return {nid: node.state for nid, node in network.nodes.items()}


def _restore_states(network: Network, states: Dict[int, NodeState]) -> None:
    """Restore node states from a snapshot."""
    for nid, state in states.items():
        network.nodes[nid].state = state


# ------------------------------------------------------------------
# Main Streamlit app
# ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Network Attack Simulator", layout="wide")
    st.title("🦠 Network Attack Simulator")

    # ---- Sidebar: simulation configuration ----
    st.sidebar.header("Simulation Configuration")

    topology = st.sidebar.selectbox(
        "Topology",
        ["Erdős–Rényi (Random)", "Star", "Barabási–Albert (Scale-Free)"],
    )
    n_nodes = st.sidebar.slider("Number of nodes", 10, 200, 50)

    if topology == "Erdős–Rényi (Random)":
        avg_degree = st.sidebar.slider("Average degree", 1.0, 15.0, 4.0, 0.5)
    elif topology == "Barabási–Albert (Scale-Free)":
        ba_m = st.sidebar.slider(
            "Edges per new node (m)", 1, min(10, n_nodes - 1), 2
        )

    vuln_lo = st.sidebar.slider("Min vulnerability", 0.0, 1.0, 0.3, 0.05)
    vuln_hi = st.sidebar.slider("Max vulnerability", 0.0, 1.0, 0.9, 0.05)
    if vuln_lo > vuln_hi:
        vuln_lo, vuln_hi = vuln_hi, vuln_lo

    remove_orphans = st.sidebar.checkbox(
        "Remove orphan nodes",
        value=False,
        help="Connect isolated (degree-0) nodes to a random neighbor "
        "so no node is unreachable. Only affects Erdős–Rényi graphs.",
    )

    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    patient_zero = st.sidebar.number_input(
        "Patient zero node ID", value=0, min_value=0, max_value=n_nodes - 1, step=1
    )
    max_epochs = st.sidebar.number_input(
        "Max epochs", value=500, min_value=1, max_value=5000, step=50
    )

    # ---- Tabs for the three main features ----
    tab_run, tab_batch, tab_replay = st.tabs(
        ["▶ Run Simulation", "📊 Batch Analysis", "🔄 Replay Infection"]
    )

    # ==================================================================
    # TAB 1 – Single simulation run & results
    # ==================================================================
    with tab_run:
        if st.button("Run Simulation", key="run_single"):
            with st.spinner("Building network & running simulation…"):
                net = _build_network(
                    topology, n_nodes, seed,
                    avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    (vuln_lo, vuln_hi),
                    remove_orphans=remove_orphans,
                )
                engine = RandomWalkEngine(seed=int(seed))
                orch = SimulationOrchestrator(net, engine)
                orch.set_patient_zero([int(patient_zero)])
                history = orch.run(max_epochs=int(max_epochs))

                # Store for replay tab
                st.session_state["last_history"] = history
                st.session_state["last_net_params"] = {
                    "topology": topology,
                    "n_nodes": n_nodes,
                    "seed": int(seed),
                    "avg_degree": avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    "ba_m": ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    "vuln_range": (vuln_lo, vuln_hi),
                    "patient_zero": int(patient_zero),
                    "max_epochs": int(max_epochs),
                }

            # Results
            epochs_full = orch.epochs_to_full_infection()
            col1, col2, col3 = st.columns(3)
            col1.metric("Nodes", net.node_count())
            col2.metric("Edges", net.edge_count())
            col3.metric(
                "Epochs to full infection",
                epochs_full if epochs_full is not None else "N/A",
            )

            st.subheader("Infection Curve")
            curve_fig = _draw_curve_fig({"Infection": orch.infection_curve()})
            st.pyplot(curve_fig)
            plt.close(curve_fig)

            st.subheader("Final Network State")
            pos = _spring_layout(net, seed=int(seed))
            net_fig = _draw_network_fig(net, pos, title="Final State")
            st.pyplot(net_fig)
            plt.close(net_fig)

            st.subheader("Epoch Details")
            rows = [
                {
                    "Epoch": r.epoch,
                    "Infected": r.infected_count,
                    "Safe": r.safe_count,
                    "Ratio": f"{r.infection_ratio:.2%}",
                    "Newly Infected": ", ".join(map(str, r.newly_infected_ids)),
                }
                for r in history
            ]
            st.dataframe(rows, use_container_width=True)

    # ==================================================================
    # TAB 2 – Batch analysis
    # ==================================================================
    with tab_batch:
        st.subheader("Batch Analysis")
        st.write(
            "Run multiple simulations across different average degrees "
            "and compute Pearson correlation between connectivity and "
            "infection speed."
        )
        degrees_input = st.text_input(
            "Average degrees (comma-separated)", "2, 4, 6, 8"
        )
        n_runs = st.number_input(
            "Runs per degree", value=20, min_value=1, max_value=500, step=5
        )

        if st.button("Run Batch", key="run_batch"):
            try:
                degrees = [float(d.strip()) for d in degrees_input.split(",")]
            except ValueError:
                st.error("Invalid degree list. Use comma-separated numbers.")
                degrees = []

            if degrees:
                analytics = AnalyticsEngine(seed=int(seed))
                degree_results: List[Tuple[float, BatchResult]] = []
                progress = st.progress(0.0)
                for idx, avg_d in enumerate(degrees):
                    factory = lambda d=avg_d: Network.random_graph(
                        n_nodes,
                        average_degree=d,
                        vulnerability_range=(vuln_lo, vuln_hi),
                        seed=None,
                    )
                    result = analytics.run_batch(
                        factory,
                        patient_zero_ids=[int(patient_zero)],
                        num_runs=int(n_runs),
                        max_epochs=int(max_epochs),
                        collect_infection_order=True,
                    )
                    degree_results.append((avg_d, result))
                    progress.progress((idx + 1) / len(degrees))

                # Summary table
                summary_rows = []
                for deg, br in degree_results:
                    summary_rows.append(
                        {
                            "Avg Degree": deg,
                            "Mean Epochs": (
                                f"{br.mean_epochs_to_full_infection:.1f}"
                                if br.mean_epochs_to_full_infection is not None
                                else "N/A"
                            ),
                            "Std Epochs": (
                                f"{br.std_epochs_to_full_infection:.1f}"
                                if br.std_epochs_to_full_infection is not None
                                else "N/A"
                            ),
                        }
                    )
                st.dataframe(summary_rows, use_container_width=True)

                # Pearson correlation
                r, p = AnalyticsEngine.degree_vs_infection_speed(degree_results)
                if r is not None:
                    st.info(f"Pearson r = {r:.3f}, p ≈ {p:.4f}")
                else:
                    st.warning("Not enough data to compute correlation.")

                # Mean infection curves
                curve_data = {
                    f"deg={deg:.1f}": br.mean_infection_curve
                    for deg, br in degree_results
                }
                fig = _draw_curve_fig(
                    curve_data, title="Mean Infection Curves by Avg Degree"
                )
                st.pyplot(fig)
                plt.close(fig)

                # ----------------------------------------------------------
                # Infection heatmap – per-degree network heatmaps showing
                # average epoch at which each node was infected.
                # ----------------------------------------------------------
                st.subheader("🔥 Infection Heatmap (Mean Infection Order)")
                st.write(
                    "Each node is colored by the average epoch at which it "
                    "was first infected across all batch runs. "
                    "**Dark purple** = infected early, **bright yellow** = "
                    "infected late."
                )
                for deg, br in degree_results:
                    if br.mean_infection_order:
                        # Build a representative network for layout
                        heatmap_net = Network.random_graph(
                            n_nodes,
                            average_degree=deg,
                            vulnerability_range=(vuln_lo, vuln_hi),
                            seed=int(seed),
                        )
                        heatmap_pos = _spring_layout(heatmap_net, seed=int(seed))
                        heatmap_fig = _draw_infection_heatmap(
                            heatmap_net, heatmap_pos,
                            br.mean_infection_order,
                            title=f"Avg Infection Order — deg={deg:.1f}",
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

    # ==================================================================
    # TAB 3 – Step-by-step replay with Play/Pause
    # ==================================================================
    with tab_replay:
        st.subheader("Step-by-Step Infection Replay")
        st.write(
            "Replay the infection spreading epoch by epoch on the network graph. "
            "**Orange** edges show paths the random walk considered; "
            "**crimson** edges show the paths it chose. "
            "Use the **Play** button to animate automatically or drag the slider manually."
        )

        if st.button("Prepare Replay", key="prepare_replay"):
            # Stop any ongoing playback when re-preparing
            st.session_state["replay_playing"] = False

            with st.spinner("Building network & running full simulation…"):
                net = _build_network(
                    topology, n_nodes, seed,
                    avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    (vuln_lo, vuln_hi),
                    remove_orphans=remove_orphans,
                )
                engine = RandomWalkEngine(seed=int(seed))
                orch = SimulationOrchestrator(net, engine)
                orch.set_patient_zero([int(patient_zero)])
                history = orch.run(max_epochs=int(max_epochs))

                pos = _spring_layout(net, seed=int(seed))

                # Build snapshots with detailed edge data via step_detailed
                replay_net = _build_network(
                    topology, n_nodes, seed,
                    avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    (vuln_lo, vuln_hi),
                    remove_orphans=remove_orphans,
                )
                replay_engine = RandomWalkEngine(seed=int(seed))
                replay_orch = SimulationOrchestrator(replay_net, replay_engine)
                replay_orch.set_patient_zero([int(patient_zero)])

                snapshots: List[Dict[int, str]] = []
                snapshots.append(_snapshot_states_str(replay_net))
                for _ in range(len(history) - 1):
                    if replay_net.infection_ratio() >= 1.0:
                        break
                    replay_orch.step_detailed()
                    snapshots.append(_snapshot_states_str(replay_net))

                st.session_state["replay_snapshots"] = snapshots
                st.session_state["replay_pos"] = pos
                # Use the detailed history from replay_orch (has edge data)
                st.session_state["replay_history"] = replay_orch.history
                st.session_state["replay_net_params"] = {
                    "topology": topology,
                    "n_nodes": n_nodes,
                    "seed": int(seed),
                    "avg_degree": avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    "ba_m": ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    "vuln_range": (vuln_lo, vuln_hi),
                    "remove_orphans": remove_orphans,
                }
                st.session_state["replay_epoch_idx"] = 0

            st.success(
                f"Replay ready — {len(snapshots)} epochs captured."
            )

        if "replay_snapshots" in st.session_state:
            snapshots = st.session_state["replay_snapshots"]
            pos = st.session_state["replay_pos"]
            history = st.session_state["replay_history"]
            params = st.session_state["replay_net_params"]

            # ---- Playback controls ----
            ctrl_cols = st.columns([1, 1, 1, 2, 1])
            with ctrl_cols[0]:
                if st.button("▶ Play", key="replay_play"):
                    st.session_state["replay_playing"] = True
            with ctrl_cols[1]:
                if st.button("⏸ Pause", key="replay_pause"):
                    st.session_state["replay_playing"] = False
            with ctrl_cols[2]:
                if st.button("⏮ Reset", key="replay_reset"):
                    st.session_state["replay_playing"] = False
                    st.session_state["replay_epoch_idx"] = 0
            with ctrl_cols[3]:
                playback_speed = st.slider(
                    "Speed (seconds per frame)",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.3,
                    step=0.05,
                    key="replay_speed",
                )
            with ctrl_cols[4]:
                loop_animation = st.checkbox("Loop", key="replay_loop")

            is_playing = st.session_state.get("replay_playing", False)

            def _on_slider_change() -> None:
                """Pause playback when the user manually drags the slider."""
                st.session_state["replay_playing"] = False
                st.session_state["replay_epoch_idx"] = st.session_state[
                    "replay_slider"
                ]

            # Sync the slider widget key with the internal epoch tracker
            # so auto-advance and manual scrub share a single source of
            # truth.  We must write replay_slider *before* the widget
            # renders, because Streamlit ignores the ``value`` parameter
            # when the key already exists in session state.  The slider
            # is always enabled; dragging it pauses playback via the
            # on_change callback (programmatic writes do not trigger it).
            current_idx = st.session_state.get("replay_epoch_idx", 0)
            st.session_state["replay_slider"] = current_idx

            epoch_idx = st.slider(
                "Epoch",
                0,
                len(snapshots) - 1,
                value=current_idx,
                key="replay_slider",
                on_change=_on_slider_change,
            )
            st.session_state["replay_epoch_idx"] = epoch_idx

            snapshot = snapshots[epoch_idx]
            rec = history[epoch_idx] if epoch_idx < len(history) else history[-1]

            # Rebuild a lightweight network just for drawing
            net = _build_network(
                params["topology"],
                params["n_nodes"],
                params["seed"],
                params.get("avg_degree"),
                params.get("ba_m"),
                params["vuln_range"],
                remove_orphans=params.get("remove_orphans", False),
            )
            _restore_states_str(net, snapshot)

            col_graph, col_info = st.columns([2, 1])
            with col_graph:
                plotly_fig = _draw_network_plotly(
                    net, pos,
                    title=f"Epoch {rec.epoch}",
                    considered_edges=rec.considered_edges,
                    chosen_edges=rec.chosen_edges,
                    newly_infected_ids=rec.newly_infected_ids,
                )
                st.plotly_chart(plotly_fig, use_container_width=True)

            with col_info:
                st.metric("Epoch", rec.epoch)
                st.metric("Infected", rec.infected_count)
                st.metric("Safe", rec.safe_count)
                st.metric("Infection Ratio", f"{rec.infection_ratio:.2%}")
                if rec.newly_infected_ids:
                    st.write(
                        f"**Newly infected:** {', '.join(map(str, rec.newly_infected_ids))}"
                    )
                if rec.considered_edges:
                    st.write(
                        f"**Edges evaluated:** {len(rec.considered_edges)}"
                    )
                if rec.chosen_edges:
                    st.write(
                        f"**Edges traversed:** {len(rec.chosen_edges)}"
                    )

            # Show infection curve up to current epoch with improvements
            full_ratios = [h.infection_ratio for h in history]
            partial_curve = full_ratios[: epoch_idx + 1]
            fig_curve = _draw_curve_fig(
                {"Infection": partial_curve},
                title=f"Infection Growth (epoch {rec.epoch})",
                highlight_epoch=epoch_idx,
                full_curve=full_ratios,
            )
            st.pyplot(fig_curve)
            plt.close(fig_curve)

            # Auto-advance when playing
            if is_playing:
                at_end = epoch_idx >= len(snapshots) - 1
                if not at_end:
                    time.sleep(playback_speed)
                    st.session_state["replay_epoch_idx"] = epoch_idx + 1
                    st.rerun()
                elif loop_animation:
                    time.sleep(playback_speed)
                    st.session_state["replay_epoch_idx"] = 0
                    st.rerun()
                else:
                    st.session_state["replay_playing"] = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_network(
    topology: str,
    n_nodes: int,
    seed: int,
    avg_degree: Optional[float] = None,
    ba_m: Optional[int] = None,
    vuln_range: Tuple[float, float] = (0.3, 0.9),
    remove_orphans: bool = False,
) -> Network:
    """Create a network from the given parameters."""
    if topology == "Erdős–Rényi (Random)":
        return Network.random_graph(
            n_nodes,
            average_degree=avg_degree or 4.0,
            vulnerability_range=vuln_range,
            seed=int(seed),
            remove_orphans=remove_orphans,
        )
    elif topology == "Star":
        return Network.star_graph(
            n_nodes, vulnerability_range=vuln_range, seed=int(seed)
        )
    else:  # Barabási–Albert
        return Network.scale_free_graph(
            n_nodes, m=ba_m or 2, vulnerability_range=vuln_range, seed=int(seed)
        )


def _snapshot_states_str(network: Network) -> Dict[int, str]:
    """Capture node states as serialisable strings."""
    return {nid: node.state.value for nid, node in network.nodes.items()}


def _restore_states_str(network: Network, states: Dict[int, str]) -> None:
    """Restore node states from string snapshot."""
    for nid, val in states.items():
        network.nodes[nid].state = NodeState(val)


if __name__ == "__main__":
    main()
