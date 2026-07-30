import random

from agents import DynaQAgent, DynaQPlusAgent, QLearningAgent
from core.map_factory import MapFactory
from core.simulation import EpisodeEndReason, Simulation
from experiments.experiment_config import ExperimentConfig


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.world = MapFactory.create(config.environment)

        if config.algorithm == QLearningAgent.name:
            self.agent = QLearningAgent(
                world=self.world,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon=config.epsilon,
            )
        elif config.algorithm == DynaQAgent.name:
            self.agent = DynaQAgent(
                world=self.world,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon=config.epsilon,
                planning_steps=config.planning_steps,
            )
        elif config.algorithm == DynaQPlusAgent.name:
            self.agent = DynaQPlusAgent(
                world=self.world,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon=config.epsilon,
                planning_steps=config.planning_steps,
                exploration_bonus=config.exploration_bonus,
            )
        else:
            raise ValueError(f"Unknown algorithm '{config.algorithm}'")

        self.simulation = Simulation(self.agent)
        self.current_step = 0
        self.step_history: list[dict] = []

    @property
    def cumulative_reward(self) -> float:
        return sum(step["reward"] for step in self.step_history)

    @property
    def finished(self) -> bool:
        return self.current_step >= self.config.total_steps

    def run(self):
        random.seed(self.config.seed)

        while not self.finished:
            if self.current_step == self.config.change_step:
                self.world.map.set_state(2)
    
            if self.current_step == self.config.change_step + self.config.change_duration:
                self.world.map.set_state(1)
    
            reward_before_step = self.simulation.total_reward
            self.simulation.step()
            step_reward = self.simulation.total_reward - reward_before_step
            self.step_history.append(
                {
                    "reward": step_reward,
                    "world_state": self.world.map.current_state,
                }
            )
            self.current_step += 1
    
            episode_finished = self.simulation.done or self.simulation.steps >= self.config.max_episode_steps
    
            if episode_finished or self.finished:
                if self.simulation.done:
                    end_reason = self.simulation.terminal_end_reason
                elif episode_finished:
                    end_reason = EpisodeEndReason.MAX_STEPS
                else:
                    end_reason = EpisodeEndReason.EXPERIMENT_END

                self.simulation.finish_episode(end_reason)
    
            if episode_finished and not self.finished:
                self.simulation.reset_episode()
