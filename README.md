# Stability-Aware Adaptation to Structural Transition Changes in Tabular Dyna-Q

This repository contains the final project for the 2026 Reinforcement Learning
course. It studies how tabular Dyna-Q agents should react when the same
state-action pair suddenly produces a different transition.

Standard deterministic Dyna-Q stores one transition for each state-action pair.
Every real observation immediately replaces the previous model entry. This is
useful after a permanent change, but transient noise can also replace the model
and subsequently be amplified by planning.

## Research question

> Can recency-weighted transition stability improve Dyna-Q's robustness to
> transient structural changes while preserving its ability to adapt to
> permanent changes?

We evaluate whether the proposed method reduces reward loss and variance during
noise without substantially reducing performance in stationary environments or
slowing recovery after permanent changes.

## Compared agents

| Agent | Description |
| --- | --- |
| `q-learning` | Model-free baseline that learns only from real interactions. |
| `dyna-q` | Learns a deterministic transition model and uses it for additional planning updates. |
| `dyna-q-plus` | Adds a time-dependent exploration bonus for actions that have not been tried recently. |
| `local-change-dyna-q` | Accumulates mismatch evidence and performs a hard local model and Q-value repair after a threshold. |
| `stability-aware-dyna-q` | Maintains multiple transition candidates per state-action pair and weights learning and planning using recency-weighted evidence. |

The initial proposal considered a hard local change detector and latent
state-level stability classes. `LocalChangeDynaQAgent` implements the hard repair
variant. The final Stability-Aware agent instead models stability per
state-action pair because transition behavior depends on both the state and the
selected action.

## Environments

The experiments use small tabular grid worlds. Each map can switch between
complete structural states according to a reproducible behavior.

| Environment | Behavior | Description |
| --- | --- | --- |
| `two-routes` | `stationary` | The initial map remains unchanged. |
| `two-routes` | `permanent` | The upper route changes permanently at step 1000. |
| `two-routes` | `temporary` | The route changes from step 1000 to 1500 and then returns. |
| `two-routes` | `noise` | Short state changes occur between steps 1000 and 1500 with probability 0.2. |
| `shortcut` | `permanent` | A shortcut opens permanently at step 1000. |

## Installation

Python 3.11 is recommended. From the repository root in Windows PowerShell:

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The optional Qt visualization additionally requires:

```powershell
python -m pip install PySide6
```

## Running an experiment

The command-line runner executes one experiment without a UI or database:

```powershell
python -m experiments.run --algorithm stability-aware-dyna-q --environment two-routes --world-behavior noise --seed 0 --change-step 1000 --change-duration 500 --steps-after-change 2000
```

All parameters and available choices are listed with:

```powershell
python -m experiments.run --help
```

## Running the complete experiment matrix

The PyExperimenter configuration is stored in
`experiments/pyexperimenter.yml`. It currently combines 50 seeds, five agents,
six additional Stability-Aware ablation settings, and five scenarios, resulting
in 2,750 executions.

Run one pending execution as a smoke test:

```powershell
python -m experiments.run_pyexperimenter --max-experiments 1
```

Run all pending executions:

```powershell
python -m experiments.run_pyexperimenter
```

Results are stored in `experiments/pyexperimenter_results.db`. The database is
excluded from Git because per-step records make it large. Final result databases
must therefore be archived separately.

## Analysis

The notebook reads the SQLite database and aggregates learning curves and
adaptation metrics across seeds:

```powershell
jupyter lab analysis/notebook/experiment_analysis.ipynb
```

Recorded raw data supports the evaluation of reward, variance, cumulative loss,
strongest performance drop, recovery duration, success rates, model mismatches,
transition stability, and hard repair events.

## Visualization

The Qt interface is independent of the experiment runner and is intended for
interactive inspection only:

```powershell
python -m ui.qt_app
```

It is not used for quantitative experiments because rendering would unnecessarily
slow down execution.

## Tests

The deterministic test suite uses Python's standard `unittest` framework:

```powershell
python -m unittest discover -s tests -v
```

It covers agent updates, learned models, both change mechanisms, world behaviors,
experiment records, and exact reproducibility of all five agents with identical
seeds.

## Reproducibility

- Every experiment stores its complete configuration and seed.
- Action selection, planning, grid-world stochasticity, and transient noise are
  seeded.
- All agents use the same 50 seeds for paired comparisons.
- Repeating the same code and configuration produces identical step and episode
  histories.
- Final databases, figures, configurations, and learned tabular model states
  should be archived together with the report artifacts.

## Project structure

```text
agents/                  Agent implementations and transition models
core/                    Grid world, map states, and simulation
experiments/             Experiment API, runners, configuration, and DB writer
analysis/notebook/       SQLite analysis notebook
ui/                      Optional Qt visualization
tests/                   Deterministic unit and integration tests
```

## Authors and provenance

- [Prerana Pandey](https://github.com/prerannaa)
- [Merlin Roechert](https://github.com/MerlinRoechert)

The project builds on a previously developed Mars Rover/GridWorld course project.
The environment representation, changing-world behaviors, experiment pipeline,
agent comparison, change-detection variant, stability-aware transition model, and
analysis were developed or substantially revised for this project. Individual
contributions are documented in the final report.

## References

- Richard S. Sutton. *Integrated Modeling and Control Based on Reinforcement
  Learning and Dynamic Programming*. NeurIPS, 1990.
  [Paper](https://papers.nips.cc/paper_files/paper/1990/file/d9fc5b73a8d78fad3d6dffe419384e70-Paper.pdf)
- Christopher J. C. H. Watkins and Peter Dayan. *Q-learning*. Machine Learning,
  1992. [Article](https://link.springer.com/article/10.1007/BF00992698)
- Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An
  Introduction*, second edition, Chapter 8.
  [Book](http://incompleteideas.net/book/the-book-2nd.html)
- [PyExperimenter](https://github.com/tornede/py_experimenter), used for
  experiment scheduling and result storage.
