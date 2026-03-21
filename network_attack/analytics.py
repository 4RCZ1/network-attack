"""Module 4: Analytics Engine.

Provides tools for running batch simulations and extracting statistical
insights from the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from network_attack.network import Network, NodeState
from network_attack.orchestrator import EpochRecord, SimulationOrchestrator
from network_attack.random_walk import RandomWalkEngine


@dataclass
class BatchResult:
    """Aggregated results from a batch of simulation runs."""

    num_runs: int
    mean_epochs_to_full_infection: Optional[float]
    std_epochs_to_full_infection: Optional[float]
    mean_infection_curve: List[float]
    individual_epochs: List[Optional[int]]


class AnalyticsEngine:
    """Run batch simulations and compute statistics."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._base_seed = seed

    def run_batch(
        self,
        network_factory: Callable[[], Network],
        patient_zero_ids: Sequence[int],
        num_runs: int = 100,
        max_epochs: int = 1000,
    ) -> BatchResult:
        """Execute *num_runs* independent simulations.

        *network_factory* must return a **fresh** ``Network`` each call
        so that every run starts from a clean state.
        """
        all_epochs: List[Optional[int]] = []
        all_curves: List[List[float]] = []

        for i in range(num_runs):
            seed = (self._base_seed + i) if self._base_seed is not None else None
            net = network_factory()
            engine = RandomWalkEngine(seed=seed)
            orch = SimulationOrchestrator(net, engine)
            orch.set_patient_zero(patient_zero_ids)
            orch.run(max_epochs=max_epochs)

            all_epochs.append(orch.epochs_to_full_infection())
            all_curves.append(orch.infection_curve())

        # Compute mean infection curve (pad shorter curves with 1.0)
        max_len = max(len(c) for c in all_curves) if all_curves else 0
        padded = np.ones((num_runs, max_len), dtype=np.float64)
        for idx, curve in enumerate(all_curves):
            padded[idx, : len(curve)] = curve
        mean_curve = padded.mean(axis=0).tolist() if max_len > 0 else []

        # Epochs statistics (only for runs that reached full infection)
        finite_epochs = [e for e in all_epochs if e is not None]
        if finite_epochs:
            mean_e: Optional[float] = float(np.mean(finite_epochs))
            std_e: Optional[float] = float(np.std(finite_epochs))
        else:
            mean_e = None
            std_e = None

        return BatchResult(
            num_runs=num_runs,
            mean_epochs_to_full_infection=mean_e,
            std_epochs_to_full_infection=std_e,
            mean_infection_curve=mean_curve,
            individual_epochs=all_epochs,
        )

    @staticmethod
    def degree_vs_infection_speed(
        results: Sequence[Tuple[float, BatchResult]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute Pearson correlation between average degree and
        mean epochs to full infection.

        *results* is a sequence of ``(average_degree, BatchResult)`` pairs.

        Returns ``(correlation, p_value)`` or ``(None, None)`` if there
        are fewer than 3 data points or correlation cannot be computed.
        """
        degrees: List[float] = []
        speeds: List[float] = []
        for deg, br in results:
            if br.mean_epochs_to_full_infection is not None:
                degrees.append(deg)
                speeds.append(br.mean_epochs_to_full_infection)

        if len(degrees) < 3:
            return None, None

        x = np.array(degrees)
        y = np.array(speeds)

        # Pearson r computed manually (avoids scipy dependency)
        x_m = x - x.mean()
        y_m = y - y.mean()
        num = (x_m * y_m).sum()
        den = np.sqrt((x_m**2).sum() * (y_m**2).sum())
        if den == 0:
            return None, None
        r = float(num / den)

        # Two-tailed p-value approximation via t-distribution
        n = len(degrees)
        t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-15))
        # Simple approximation using the survival function of t-dist
        # (good enough for exploratory analysis)
        from math import erfc, sqrt

        p_value = float(erfc(abs(t_stat) / sqrt(2)))

        return r, p_value
