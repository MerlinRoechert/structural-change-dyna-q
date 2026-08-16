import logging
import random

from agents import DynaQAgent, DynaQPlusAgent, QLearningAgent, StabilityAwareDynaQAgent
from core.map_factory import MapFactory
from core.simulation import EpisodeEndReason, Simulation
from core.world_state_behavior import WorldStateBehaviorFactory
from experiments.experiment_config import ExperimentConfig


class Experiment:
    def __init__(self, config: ExperimentConfig):
        state_behavior = WorldStateBehaviorFactory.create(
            world_behavior=config.world_behavior,
            change_step=config.change_step,
            change_duration=config.change_duration,
            seed=config.seed,
        )
        self.config = config
        self.world = MapFactory.create(
            world_id=config.environment,
            state_behavior=state_behavior,
            seed=config.seed,
        )

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
        elif config.algorithm == StabilityAwareDynaQAgent.name:
            self.agent = StabilityAwareDynaQAgent(
                world=self.world,
                learning_rate=config.learning_rate,
                discount_factor=config.discount_factor,
                epsilon=config.epsilon,
                planning_steps=config.planning_steps,
                evidence_decay=config.evidence_decay,
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

    def run(self, logger: logging.Logger):
        random.seed(self.config.seed)
        progress_steps = {
            round(self.config.total_steps * fraction)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        }

        while not self.finished:
            previous_state = self.world.map.current_state
            state_changed = self.world.update_state(self.current_step)
            if state_changed and self.world.state_behavior.log_state_changes:
                logger.info(
                    "World changed: step=%s, state=%s->%s",
                    self.current_step,
                    previous_state,
                    self.world.map.current_state,
                )
    
            reward_before_step = self.simulation.total_reward
            self.simulation.step()
            step_reward = self.simulation.total_reward - reward_before_step
            state_row, state_column = self.agent.last_state
            next_state_row, next_state_column = self.agent.last_next_state
            step_record = {
                "state_row": state_row,
                "state_column": state_column,
                "action": self.agent.last_action.value,
                "next_state_row": next_state_row,
                "next_state_column": next_state_column,
                "reward": step_reward,
                "terminated": self.agent.terminated,
                "world_state": self.world.map.current_state,
                "model_mismatch": None,
                "stability_score": None,
                "observed_transition_stability": None,
                "candidate_count": None,
            }
            if isinstance(self.agent, StabilityAwareDynaQAgent):
                step_record["model_mismatch"] = self.agent.model_mismatch
                step_record["stability_score"] = self.agent.stability_score
                step_record["observed_transition_stability"] = (
                    self.agent.observed_transition_stability
                )
                step_record["candidate_count"] = self.agent.candidate_count

            self.step_history.append(step_record)
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

            if self.current_step in progress_steps:
                logger.info(
                    "Experiment progress: step=%s/%s, episodes=%s, reward=%s",
                    self.current_step,
                    self.config.total_steps,
                    len(self.simulation.episode_summaries),
                    self.cumulative_reward,
                )
