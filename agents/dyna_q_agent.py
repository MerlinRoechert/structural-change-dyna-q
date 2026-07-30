import random

from agents.q_learning_agent import QLearningAgent
from core.action import Action
from core.grid_world import GridWorld


class ModelTransition:
    def __init__(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ):
        self.next_state = next_state
        self.reward = reward
        self.terminated = terminated


class DynaQAgent(QLearningAgent):
    name = "dyna-q"
    DEFAULT_PLANNING_STEPS = 10

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DEFAULT_PLANNING_STEPS,
    ):
        if planning_steps < 0:
            raise ValueError("planning_steps must not be negative")

        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
        )
        self.planning_steps = planning_steps
        self.model: dict[
            tuple[tuple[int, int], Action],
            ModelTransition,
        ] = {}

    def learn(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        super().learn(
            current_state=current_state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )

        self._update_model(
            current_state=current_state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )
        self._run_planning()

    def reset_learning(self) -> None:
        super().reset_learning()
        self.model.clear()

    def _run_planning(self) -> None:
        model_entries = tuple(self.model.items())

        for _ in range(self.planning_steps):
            (state, action), transition = random.choice(model_entries)
            super().learn(
                current_state=state,
                action=action,
                next_state=transition.next_state,
                reward=self._planning_reward(transition),
                terminated=transition.terminated,
            )

    def _update_model(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self.model[(current_state, action)] = ModelTransition(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )

    def _planning_reward(self, transition: ModelTransition) -> float:
        return transition.reward
