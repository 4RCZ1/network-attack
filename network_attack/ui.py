"""Streamlit UI for managing network attack simulations.

Launch with:
    streamlit run network_attack/ui.py
"""

from __future__ import annotations

import io
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
COLOR_INFECTED = "#e74c3c"


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
        color = COLOR_INFECTED if node.state == NodeState.INFECTED else COLOR_SAFE
        size = 40 + 160 * node.vulnerability
        ax.scatter(
            x, y, s=size, c=color, edgecolors="black", linewidths=0.5, zorder=2
        )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _draw_curve_fig(
    curves: Dict[str, List[float]], title: str = "Infection Growth"
) -> plt.Figure:
    """Return a matplotlib Figure with infection curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in curves.items():
        ax.plot(range(len(data)), data, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Infection Ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend()
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

    # ==================================================================
    # TAB 3 – Step-by-step replay
    # ==================================================================
    with tab_replay:
        st.subheader("Step-by-Step Infection Replay")
        st.write(
            "Replay the infection spreading epoch by epoch on the network graph."
        )

        if st.button("Prepare Replay", key="prepare_replay"):
            with st.spinner("Building network & running full simulation…"):
                net = _build_network(
                    topology, n_nodes, seed,
                    avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    (vuln_lo, vuln_hi),
                )
                engine = RandomWalkEngine(seed=int(seed))
                orch = SimulationOrchestrator(net, engine)
                orch.set_patient_zero([int(patient_zero)])
                history = orch.run(max_epochs=int(max_epochs))

                pos = _spring_layout(net, seed=int(seed))

                # Build snapshots: re-run to capture state at each epoch
                replay_net = _build_network(
                    topology, n_nodes, seed,
                    avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    (vuln_lo, vuln_hi),
                )
                replay_engine = RandomWalkEngine(seed=int(seed))
                replay_orch = SimulationOrchestrator(replay_net, replay_engine)
                replay_orch.set_patient_zero([int(patient_zero)])

                snapshots: List[Dict[int, str]] = []
                snapshots.append(_snapshot_states_str(replay_net))
                for _ in range(len(history) - 1):
                    if replay_net.infection_ratio() >= 1.0:
                        break
                    replay_orch.step()
                    snapshots.append(_snapshot_states_str(replay_net))

                st.session_state["replay_snapshots"] = snapshots
                st.session_state["replay_pos"] = pos
                st.session_state["replay_history"] = history
                st.session_state["replay_net_params"] = {
                    "topology": topology,
                    "n_nodes": n_nodes,
                    "seed": int(seed),
                    "avg_degree": avg_degree if topology == "Erdős–Rényi (Random)" else None,
                    "ba_m": ba_m if topology == "Barabási–Albert (Scale-Free)" else None,
                    "vuln_range": (vuln_lo, vuln_hi),
                }

            st.success(
                f"Replay ready — {len(snapshots)} epochs captured."
            )

        if "replay_snapshots" in st.session_state:
            snapshots = st.session_state["replay_snapshots"]
            pos = st.session_state["replay_pos"]
            history = st.session_state["replay_history"]
            params = st.session_state["replay_net_params"]

            epoch_idx = st.slider(
                "Epoch", 0, len(snapshots) - 1, 0, key="replay_slider"
            )

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
            )
            _restore_states_str(net, snapshot)

            col_graph, col_info = st.columns([2, 1])
            with col_graph:
                fig = _draw_network_fig(
                    net, pos, title=f"Epoch {rec.epoch}"
                )
                st.pyplot(fig)
                plt.close(fig)

            with col_info:
                st.metric("Epoch", rec.epoch)
                st.metric("Infected", rec.infected_count)
                st.metric("Safe", rec.safe_count)
                st.metric("Infection Ratio", f"{rec.infection_ratio:.2%}")
                if rec.newly_infected_ids:
                    st.write(
                        f"**Newly infected:** {', '.join(map(str, rec.newly_infected_ids))}"
                    )

            # Show infection curve up to current epoch
            partial_curve = [
                h.infection_ratio for h in history[: epoch_idx + 1]
            ]
            fig_curve = _draw_curve_fig(
                {"Infection": partial_curve},
                title=f"Infection Growth (up to epoch {rec.epoch})",
            )
            st.pyplot(fig_curve)
            plt.close(fig_curve)


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
) -> Network:
    """Create a network from the given parameters."""
    if topology == "Erdős–Rényi (Random)":
        return Network.random_graph(
            n_nodes,
            average_degree=avg_degree or 4.0,
            vulnerability_range=vuln_range,
            seed=int(seed),
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
