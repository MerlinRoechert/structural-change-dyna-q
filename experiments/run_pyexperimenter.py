import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.experiment import Experiment
from experiments.experiment_config import ExperimentConfig

if TYPE_CHECKING:
    from py_experimenter.result_processor import ResultProcessor


CONFIG_PATH = Path(__file__).with_name("pyexperimenter.yml")


def create_experiment_config(parameters: dict) -> ExperimentConfig:
    return ExperimentConfig(
        algorithm=parameters["algorithm"],
        environment=parameters["environment"],
        seed=int(parameters["seed"]),
        learning_rate=float(parameters["learning_rate"]),
        discount_factor=float(parameters["discount_factor"]),
        epsilon=float(parameters["epsilon"]),
        planning_steps=int(parameters["planning_steps"]),
        exploration_bonus=float(parameters["exploration_bonus"]),
        change_step=int(parameters["change_step"]),
        change_duration=int(parameters["change_duration"]),
        steps_after_change=int(parameters["steps_after_change"]),
        max_episode_steps=int(parameters["max_episode_steps"]),
    )


def iterate_step_records(experiment: Experiment) -> Iterator[dict]:
    step_index = 0

    for episode in experiment.simulation.episode_summaries:
        for _ in range(episode["steps"]):
            step = experiment.step_history[step_index]

            yield {
                "global_step": step_index + 1,
                "reward": step["reward"],
                "world_state": step["world_state"],
                "episode": episode["episode"],
            }
            step_index += 1


def iterate_episode_records(experiment: Experiment) -> Iterator[dict]:
    end_step = 0

    for episode in experiment.simulation.episode_summaries:
        end_step += episode["steps"]
        yield {
            "episode": episode["episode"],
            "end_step": end_step,
            "end_reason": episode["end_reason"],
        }


def run_experiment(
    parameters: dict,
    result_processor: "ResultProcessor",
    custom_config: dict,
) -> None:
    experiment = Experiment(create_experiment_config(parameters))
    experiment.run()

    for step_record in iterate_step_records(experiment):
        result_processor.process_logs({"steps": step_record})

    for episode_record in iterate_episode_records(experiment):
        result_processor.process_logs({"episodes": episode_record})


def main() -> None:
    from py_experimenter.experimenter import PyExperimenter

    parser = argparse.ArgumentParser(description="Run experiments with PyExperimenter")
    parser.add_argument("--max-experiments", type=int, default=-1)
    arguments = parser.parse_args()

    experimenter = PyExperimenter(
        experiment_configuration_file_path=str(CONFIG_PATH),
        use_codecarbon=False,
    )
    experimenter.fill_table_from_config()
    experimenter.execute(
        run_experiment,
        max_experiments=arguments.max_experiments,
    )


if __name__ == "__main__":
    main()
