import argparse
from collections.abc import Iterator
from pathlib import Path

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from experiments.experiment import Experiment
from experiments.experiment_config import ExperimentConfig


CONFIG_PATH = Path(__file__).with_name("pyexperimenter.yml")
DATABASE_PROGRESS_INTERVAL = 500


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
        initial_stability=float(parameters["initial_stability"]),
        stability_increase=float(parameters["stability_increase"]),
        evidence_gain=float(parameters["evidence_gain"]),
        change_evidence_decay=float(parameters["change_evidence_decay"]),
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
                "stability_score": step["stability_score"],
                "change_evidence": step["change_evidence"],
                "change_probability": step["change_probability"],
                "change_detected": step["change_detected"],
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
    result_processor: ResultProcessor,
    custom_config: dict,
) -> None:
    logger = result_processor.logger
    config = create_experiment_config(parameters)
    logger.info(
        "Starting experiment: algorithm=%s, environment=%s, seed=%s, steps=%s",
        config.algorithm,
        config.environment,
        config.seed,
        config.total_steps,
    )

    experiment = Experiment(config)
    experiment.run(logger)

    episode_count = len(experiment.simulation.episode_summaries)
    logger.info(
        "Simulation finished: episodes=%s, reward=%s",
        episode_count,
        experiment.cumulative_reward,
    )

    for saved_steps, step_record in enumerate(
        iterate_step_records(experiment),
        start=1,
    ):
        result_processor.process_logs({"steps": step_record})
        if (
            saved_steps % DATABASE_PROGRESS_INTERVAL == 0
            or saved_steps == experiment.current_step
        ):
            logger.info(
                "Saved steps to database: %s/%s",
                saved_steps,
                experiment.current_step,
            )

    for episode_record in iterate_episode_records(experiment):
        result_processor.process_logs({"episodes": episode_record})

    change_detections = sum(
        step["change_detected"] is True
        for step in experiment.step_history
    )
    logger.info(
        "Experiment finished: episodes=%s, change_detections=%s",
        episode_count,
        change_detections,
    )


def main() -> None:
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
