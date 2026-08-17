import argparse
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from experiments.batch_log_writer import BatchLogWriter
from experiments.experiment import Experiment
from experiments.experiment_config import ExperimentConfig


CONFIG_PATH = Path(__file__).with_name("pyexperimenter.yml")


def create_experiment_config(parameters: dict) -> ExperimentConfig:
    change_step = parameters["change_step"]
    if change_step is not None:
        change_step = int(change_step)

    change_duration = parameters["change_duration"]
    if change_duration is not None:
        change_duration = int(change_duration)

    return ExperimentConfig(
        algorithm=parameters["algorithm"],
        environment=parameters["environment"],
        world_behavior=parameters["world_behavior"],
        seed=int(parameters["seed"]),
        learning_rate=float(parameters["learning_rate"]),
        discount_factor=float(parameters["discount_factor"]),
        epsilon=float(parameters["epsilon"]),
        planning_steps=int(parameters["planning_steps"]),
        exploration_bonus=float(parameters["exploration_bonus"]),
        evidence_decay=float(parameters["evidence_decay"]),
        confidence_rate=float(parameters["confidence_rate"]),
        evidence_tolerance=float(parameters["evidence_tolerance"]),
        repair_threshold=float(parameters["repair_threshold"]),
        initial_confidence=float(parameters["initial_confidence"]),
        change_step=change_step,
        change_duration=change_duration,
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
                "state_row": step["state_row"],
                "state_column": step["state_column"],
                "action": step["action"],
                "next_state_row": step["next_state_row"],
                "next_state_column": step["next_state_column"],
                "reward": step["reward"],
                "terminated": step["terminated"],
                "world_state": step["world_state"],
                "episode": episode["episode"],
                "model_mismatch": step["model_mismatch"],
                "stability_score": step["stability_score"],
                "observed_transition_stability": step[
                    "observed_transition_stability"
                ],
                "candidate_count": step["candidate_count"],
                "model_confidence": step["model_confidence"],
                "change_evidence": step["change_evidence"],
                "repair_triggered": step["repair_triggered"],
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
        "Starting experiment: algorithm=%s, environment=%s, behavior=%s, "
        "seed=%s, steps=%s",
        config.algorithm,
        config.environment,
        config.world_behavior,
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

    log_writer = BatchLogWriter(result_processor)
    log_writer.add_records("steps", iterate_step_records(experiment))
    log_writer.add_records("episodes", iterate_episode_records(experiment))

    logger.info("Saving %s log records to database", log_writer.record_count)
    saving_started_at = perf_counter()
    log_writer.write()
    saving_duration = perf_counter() - saving_started_at
    logger.info(
        "Saved %s log records in %.3f seconds",
        log_writer.record_count,
        saving_duration,
    )

    logger.info("Experiment finished: episodes=%s", episode_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments with PyExperimenter")
    parser.add_argument("--max-experiments", type=int, default=-1)
    arguments = parser.parse_args()

    experimenter = PyExperimenter(
        experiment_configuration_file_path=str(CONFIG_PATH),
        use_codecarbon=False,
    )
    scenarios = [
        dict(scenario)
        for scenario in experimenter.config.custom_configuration.custom_values[
            "scenarios"
        ]
    ]
    keyfields = experimenter.config.database_configuration.keyfields
    shared_parameters = {
        name: keyfield.values
        for name, keyfield in keyfields.items()
        if keyfield.values
    }
    experimenter.fill_table_from_combination(
        fixed_parameter_combinations=scenarios,
        parameters=shared_parameters,
    )
    experimenter.execute(
        run_experiment,
        max_experiments=arguments.max_experiments,
    )


if __name__ == "__main__":
    main()
