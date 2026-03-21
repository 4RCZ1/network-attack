"""Module 3: Simulation Orchestrator (Time & State Management).

Manages the flow of a single simulation run: initialization, the main
time-step loop, and per-epoch statistics logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from network_attack.network import Network, NodeState
from network_attack.random_walk import RandomWalkEngine


@dataclass
class EpochRecord:
    """Statistics snapshot for a single simulation epoch."""

    epoch: int
    total_nodes: int
    infected_count: int
    safe_count: int
    infection_ratio: float
    newly_infected_ids: List[int]


class SimulationOrchestrator:
    """Orchestrate a single simulation run.

    Attributes:
        network: The network being simulated.
        engine: The random-walk engine used for propagation.
        history: List of ``EpochRecord`` entries (one per epoch).
        current_epoch: The current time step.
    """

    def __init__(
        self,
        network: Network,
        engine: Optional[RandomWalkEngine] = None,
    ) -> None:
        self.network = network
        self.engine = engine or RandomWalkEngine()
        self.history: List[EpochRecord] = []
        self.current_epoch: int = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def set_patient_zero(self, node_ids: Sequence[int]) -> None:
        """Mark the given nodes as the initial infection cluster.

        Resets the simulation state (epoch counter and history).
        """
        self.network.reset()
        self.current_epoch = 0
        self.history.clear()

        for nid in node_ids:
            if nid not in self.network.nodes:
                raise ValueError(f"Node {nid} does not exist in the network")
            self.network.nodes[nid].state = NodeState.INFECTED

        self._record_epoch(newly_infected_ids=list(node_ids))

    # ------------------------------------------------------------------
    # Simulation loop helpers
    # ------------------------------------------------------------------

    def _record_epoch(self, newly_infected_ids: List[int]) -> None:
        total = self.network.node_count()
        infected = len(self.network.get_infected_nodes())
        self.history.append(
            EpochRecord(
                epoch=self.current_epoch,
                total_nodes=total,
                infected_count=infected,
                safe_count=total - infected,
                infection_ratio=self.network.infection_ratio(),
                newly_infected_ids=newly_infected_ids,
            )
        )

    def step(self) -> EpochRecord:
        """Advance the simulation by one epoch.

        Returns the ``EpochRecord`` for the new epoch.
        """
        self.current_epoch += 1
        newly = self.engine.step_all_infected(self.network)
        self._record_epoch([n.node_id for n in newly])
        return self.history[-1]

    def run(self, max_epochs: int = 1000) -> List[EpochRecord]:
        """Run the simulation until the network is fully infected or
        *max_epochs* is reached.

        Returns the complete history.
        """
        for _ in range(max_epochs):
            if self.network.infection_ratio() >= 1.0:
                break
            self.step()
        return self.history

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def infection_curve(self) -> List[float]:
        """Return the infection ratio at each recorded epoch."""
        return [rec.infection_ratio for rec in self.history]

    def epochs_to_full_infection(self) -> Optional[int]:
        """Return the epoch at which 100% infection was first reached,
        or ``None`` if the network was never fully infected."""
        for rec in self.history:
            if rec.infection_ratio >= 1.0:
                return rec.epoch
        return None
