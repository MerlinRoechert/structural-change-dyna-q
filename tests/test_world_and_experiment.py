import logging
import unittest

from agents import (
    DynaQAgent,
    DynaQPlusAgent,
    LocalChangeDynaQAgent,
    QLearningAgent,
    StabilityAwareDynaQAgent,
)
from core.map_factory import MapFactory
from core.world_object import Empty, Wall
from core.world_state_behavior import (
    ScheduledWorldStateBehavior,
    StaticWorldStateBehavior,
    TransientNoiseWorldStateBehavior,
)
from experiments.experiment import Experiment
from experiments.experiment_config import ExperimentConfig


LOGGER = logging.getLogger("tests")
LOGGER.addHandler(logging.NullHandler())


class WorldBehaviorTests(unittest.TestCase):
    def test_stationary_world_never_changes_state(self):
        world = MapFactory.create(
            "two-routes",
            StaticWorldStateBehavior(),
            seed=0,
        )

        for step in range(20):
            world.update_state(step)

        self.assertEqual(world.map.current_state, 1)

    def test_permanent_change_happens_at_configured_step(self):
        world = MapFactory.create(
            "two-routes",
            ScheduledWorldStateBehavior(change_step=5, change_duration=None),
            seed=0,
        )

        world.update_state(4)
        self.assertEqual(world.map.current_state, 1)
        world.update_state(5)
        self.assertEqual(world.map.current_state, 2)
        world.update_state(6)
        self.assertEqual(world.map.current_state, 2)

    def test_temporary_change_returns_after_duration(self):
        world = MapFactory.create(
            "two-routes",
            ScheduledWorldStateBehavior(change_step=5, change_duration=3),
            seed=0,
        )

        world.update_state(5)
        self.assertEqual(world.map.current_state, 2)
        world.update_state(7)
        self.assertEqual(world.map.current_state, 2)
        world.update_state(8)
        self.assertEqual(world.map.current_state, 1)

    def test_noise_sequence_is_reproducible_with_same_seed(self):
        first_world = MapFactory.create(
            "two-routes",
            TransientNoiseWorldStateBehavior(5, 30, seed=7),
            seed=7,
        )
        second_world = MapFactory.create(
            "two-routes",
            TransientNoiseWorldStateBehavior(5, 30, seed=7),
            seed=7,
        )

        first_states = []
        second_states = []
        for step in range(40):
            first_world.update_state(step)
            second_world.update_state(step)
            first_states.append(first_world.map.current_state)
            second_states.append(second_world.map.current_state)

        self.assertEqual(first_states, second_states)
        self.assertIn(2, first_states)
        self.assertTrue(all(state == 1 for state in first_states[:5]))
        self.assertTrue(all(state == 1 for state in first_states[35:]))

    def test_factory_returns_independent_worlds(self):
        first_world = MapFactory.create(
            "two-routes",
            StaticWorldStateBehavior(),
            seed=0,
        )
        second_world = MapFactory.create(
            "two-routes",
            StaticWorldStateBehavior(),
            seed=0,
        )
        self.assertIsInstance(second_world.get_cell((1, 2)), Empty)

        first_world.set_cell((1, 2), Wall())

        self.assertIsInstance(first_world.get_cell((1, 2)), Wall)
        self.assertIsInstance(second_world.get_cell((1, 2)), Empty)


class ExperimentTests(unittest.TestCase):
    ALGORITHMS = (
        QLearningAgent.name,
        DynaQAgent.name,
        DynaQPlusAgent.name,
        LocalChangeDynaQAgent.name,
        StabilityAwareDynaQAgent.name,
    )

    def test_total_steps_includes_steps_before_and_after_change(self):
        config = ExperimentConfig(change_step=20, steps_after_change=30)

        self.assertEqual(config.total_steps, 50)

    def test_experiment_records_every_step_and_episode_boundary(self):
        experiment = Experiment(
            ExperimentConfig(
                algorithm=DynaQAgent.name,
                steps_after_change=75,
                max_episode_steps=20,
            )
        )

        experiment.run(LOGGER)

        self.assertEqual(experiment.current_step, 75)
        self.assertEqual(len(experiment.step_history), 75)
        self.assertEqual(
            sum(
                episode["steps"]
                for episode in experiment.simulation.episode_summaries
            ),
            75,
        )

    def test_same_seed_reproduces_all_agent_histories(self):
        for algorithm in self.ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                config = ExperimentConfig(
                    algorithm=algorithm,
                    world_behavior="noise",
                    seed=7,
                    change_step=20,
                    change_duration=10,
                    steps_after_change=30,
                    max_episode_steps=25,
                )
                first_experiment = Experiment(config)
                second_experiment = Experiment(config)

                first_experiment.run(LOGGER)
                second_experiment.run(LOGGER)

                self.assertEqual(
                    first_experiment.step_history,
                    second_experiment.step_history,
                )
                self.assertEqual(
                    first_experiment.simulation.episode_summaries,
                    second_experiment.simulation.episode_summaries,
                )

    def test_diagnostic_fields_are_only_filled_by_relevant_agents(self):
        baseline = Experiment(
            ExperimentConfig(
                algorithm=DynaQAgent.name,
                steps_after_change=10,
            )
        )
        local_change = Experiment(
            ExperimentConfig(
                algorithm=LocalChangeDynaQAgent.name,
                steps_after_change=10,
            )
        )
        stability_aware = Experiment(
            ExperimentConfig(
                algorithm=StabilityAwareDynaQAgent.name,
                steps_after_change=10,
            )
        )

        for experiment in (baseline, local_change, stability_aware):
            experiment.run(LOGGER)

        baseline_step = baseline.step_history[0]
        local_change_step = local_change.step_history[0]
        stability_step = stability_aware.step_history[0]

        self.assertIsNone(baseline_step["model_mismatch"])
        self.assertIsNone(baseline_step["model_confidence"])
        self.assertIsNone(baseline_step["stability_score"])

        self.assertIsNotNone(local_change_step["model_mismatch"])
        self.assertIsNotNone(local_change_step["model_confidence"])
        self.assertIsNone(local_change_step["stability_score"])

        self.assertIsNotNone(stability_step["model_mismatch"])
        self.assertIsNotNone(stability_step["stability_score"])
        self.assertIsNone(stability_step["model_confidence"])


if __name__ == "__main__":
    unittest.main()
