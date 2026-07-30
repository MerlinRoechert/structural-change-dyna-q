import argparse

from agents import DynaQAgent, DynaQPlusAgent, QLearningAgent
from core.map_factory import WorldId
from experiments.experiment import Experiment
from experiments.experiment_config import ExperimentConfig


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local RL experiment")
    parser.add_argument(
        "--algorithm",
        choices=[
            QLearningAgent.name,
            DynaQAgent.name,
            DynaQPlusAgent.name,
        ],
        default=QLearningAgent.name,
    )
    parser.add_argument(
        "--environment",
        choices=[world_id.value for world_id in WorldId],
        default=WorldId.TWO_ROUTES.value,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--planning-steps", type=int, default=10)
    parser.add_argument("--exploration-bonus", type=float, default=0.001)
    parser.add_argument("--change-step", type=int)
    parser.add_argument("--change-duration", type=int)
    parser.add_argument("--steps-after-change", type=int, default=1_000)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    return parser


def main():
    arguments = create_argument_parser().parse_args()
    config = ExperimentConfig(
        algorithm=arguments.algorithm,
        environment=arguments.environment,
        seed=arguments.seed,
        learning_rate=arguments.learning_rate,
        discount_factor=arguments.discount_factor,
        epsilon=arguments.epsilon,
        planning_steps=arguments.planning_steps,
        exploration_bonus=arguments.exploration_bonus,
        change_step=arguments.change_step,
        change_duration=arguments.change_duration,
        steps_after_change=arguments.steps_after_change,
        max_episode_steps=arguments.max_episode_steps,
    )

    experiment = Experiment(config)
    experiment.run()

    print(f"Steps: {experiment.current_step}")
    print(f"Episodes: {len(experiment.simulation.episode_summaries)}")
    print(f"Cumulative reward: {experiment.cumulative_reward}")


if __name__ == "__main__":
    main()
