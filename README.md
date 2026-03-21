# Attack Propagation in Complex Networks Using Biased Random Walks

Simulation of attack propagation through network graphs using biased random walks, where the jump probability to a neighbor is proportional to its vulnerability index.

## Architecture

The project is split into five decoupled modules:

| Module | File | Responsibility |
|--------|------|----------------|
| **Data Representation** | `network.py` | `Node` / `Network` classes, adjacency list, graph generators (Erdős–Rényi, star, Barabási–Albert) |
| **Random Walk Engine** | `random_walk.py` | Vulnerability-weighted probability vectors, biased stochastic target selection |
| **Simulation Orchestrator** | `orchestrator.py` | Patient-zero initialization, discrete time-step loop, per-epoch statistics logging |
| **Analytics Engine** | `analytics.py` | Batch simulations (N runs per topology), Pearson correlation between average degree and infection speed |
| **Visualization** | `visualization.py` | Force-directed graph rendering, infection-growth curves, interactive step-by-step viewer (matplotlib) |

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

`main.py` runs three demos:

1. **Single run** — scale-free graph (50 nodes), epoch-by-epoch infection printout.
2. **Batch analysis** — Erdős–Rényi graphs with average degrees 2 / 4 / 6 / 8; prints mean epochs to full infection and Pearson *r*.
3. **Defense strategy** — compares baseline propagation vs. a network with low-vulnerability "firewall" nodes.

## Usage Example

```python
from network_attack import Network, RandomWalkEngine, SimulationOrchestrator

net = Network.scale_free_graph(100, m=2, seed=7)
orch = SimulationOrchestrator(net, RandomWalkEngine(seed=42))
orch.set_patient_zero([0])
orch.run(max_epochs=200)
print(f"Full infection at epoch {orch.epochs_to_full_infection()}")
```

Simulate defenses by lowering vulnerability on selected nodes:

```python
net.nodes[10].vulnerability = 0.01   # firewall / patched server
```

## Running Tests

```bash
python -m pytest tests/ -v
```

39 tests cover modules 1–4 (network, random walk, orchestrator, analytics).

## Possible Improvements

The items below are missing or underdeveloped relative to the full project specification (*propagacja ataku w wybranej sieci z wykorzystaniem spacerów losowych*):

1. **Formal methodology write-up** — the random walk model and its mathematical properties (expected hitting times, spectral gap relationship) are not documented outside of code docstrings.
2. **Systematic topology comparison** — although three generators exist, there is no automated script that runs all topologies side-by-side and exports a comparison table or plot (infection curves overlaid per topology type).
3. **Additional topologies** — ring / lattice and Watts–Strogatz small-world graphs are not yet implemented; adding them would broaden the topology coverage.
4. **Edge-level weights** — propagation bias currently depends only on node vulnerability; the specification could be extended so that individual link weights (e.g., bandwidth, trust level) also influence jump probability.
5. **Recovery / SIR model** — nodes can only transition from *safe* → *infected*; there is no *recovered* state, so SIR/SIS epidemic dynamics cannot be modeled.
6. **Result export** — simulation results are kept in memory only; exporting epoch histories to CSV or JSON would improve reproducibility and allow external analysis tools.
7. **CLI interface** — all parameters (graph size, topology, seed, defense budget) must be changed in source code; a `click` or `argparse` CLI would make experimentation faster.
8. **Visualization tests** — Module 5 is excluded from the test suite due to GUI dependencies; headless / image-comparison tests could be added.
