# Analysis Plan

This document describes what we expect from our agent and how we evaluate the
final experiments.

## Research Question

Can a recency-weighted transition model make Dyna-Q more robust to temporary
structural changes without making it worse at permanent changes?

Our Stability-Aware Dyna-Q agent stores multiple observed transitions for each
state-action pair. Recent observations receive more weight through evidence
decay.

## Development and Final Runs

- Seeds 0-49 were used while developing the agent and choosing its parameters.
- Seeds 50-99 are used for the final evaluation.
- The final Stability-Aware agent uses `evidence_decay = 0.9`.
- Other decay values are only used to show how decay changes the behavior.
- All agents use the same seeds so their results can be compared directly.

We do not change the algorithm or choose another decay after looking at the
results from seeds 50-99.

## Expectations

### 1. Noise

In `two-routes`, short changes make the known upper route worse. Stability-Aware
Dyna-Q should achieve a higher average reward than Dyna-Q because older
transitions should not be replaced immediately by every short disturbance.

In `shortcut`, short changes temporarily open a beneficial route. We do not
expect Stability-Aware Dyna-Q to be better in this case. The scenario shows
whether robustness against short changes also makes the agent slower to use
short-lived opportunities. It is evaluated without a directional decision.

### 2. Permanent Change

After a permanent change, Stability-Aware Dyna-Q should still adapt. Its average
reward after the change should be no more than 0.05 per step below Dyna-Q.
This is checked separately in `two-routes` and `shortcut`.

### 3. Stationary World

Without a change, the additional transition model should not noticeably reduce
performance. The average reward should be no more than 0.05 per step below
Dyna-Q. This is checked separately in `two-routes` and `shortcut`.

### 4. Temporary Change

When the original world returns, the stored transition candidates should help
the agent reuse its previous knowledge. We therefore expect a higher average
reward than Dyna-Q during the first 500 steps after the return. This is checked
separately in `two-routes` and `shortcut`.

For `shortcut`, we also report the reward while the shortcut is open. This
shows whether retaining the old transition model slows down adaptation to a
temporary beneficial change.

## Evaluation

For every expectation, we first calculate one result per seed. We then compare
Stability-Aware Dyna-Q and Dyna-Q using the same seed.

We report:

- the average result of both agents,
- the average difference between them,
- a 95% bootstrap confidence interval for this difference,
- the median and interquartile range across seeds.

An improvement is supported when the complete confidence interval of the
difference is above zero. For stationary and permanent worlds, our agent is
considered not noticeably worse when the interval stays above -0.05.

Dyna-Q is the main comparison because our agent extends it. Dyna-Q+, Q-learning,
and Local-Change Dyna-Q are also shown to provide additional context.

Results are shown separately for each environment. We do not combine worlds if
that would hide poor performance in one of them.

## Metrics

The main metric is the average raw reward per step in the relevant phase:

- noise: all steps during the noise phase,
- permanent: all steps after the change,
- stationary: the complete run,
- temporary: the first 500 steps after the world returns.

The reward during the temporary change is additionally reported for
`shortcut`, but it does not replace the predefined return metric.

Rolling reward curves help visualize learning and adaptation, but they are not
used as separate statistical tests for every step.

Additional values such as cumulative reward, recovery time, episode length,
model mismatches, transition stability, and candidate count help explain the
results.

## Final Configuration

The experiments use the parameters from `experiments/pyexperimenter.yml`. Every
combination of the following environments and world behaviors is evaluated:

- environments: `two-routes`, `shortcut`
- world behaviors: `stationary`, `noise`, `temporary`, `permanent`

This results in eight scenarios. All scenarios use the same five agents and
seeds 50-99.

If an expectation is not supported, we report that result instead of changing
the method using the final seeds.
