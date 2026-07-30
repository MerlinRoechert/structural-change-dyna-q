import math

from agents.dyna_q_agent import DynaQAgent, ModelTransition
from core.action import Action
from core.grid_world import GridWorld


class DynaQPlusTransition(ModelTransition):
    def __init__(
        self,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
        last_tried_step: int,
    ):
        super().__init__(next_state, reward, terminated)
        self.last_tried_step = last_tried_step


class DynaQPlusAgent(DynaQAgent):
    name = "dyna-q-plus"
    DEFAULT_EXPLORATION_BONUS = 0.001

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        planning_steps: int = DynaQAgent.DEFAULT_PLANNING_STEPS,
        exploration_bonus: float = DEFAULT_EXPLORATION_BONUS,
    ):
        super().__init__(
            world=world,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            planning_steps=planning_steps,
        )
        self.exploration_bonus = exploration_bonus
        self.time_step = 0

    def learn(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self.time_step += 1
        super().learn(
            current_state=current_state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
        )

    def reset_learning(self) -> None:
        super().reset_learning()
        self.time_step = 0

    def _update_model(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        if not any(state == current_state for state, _ in self.model):
            for possible_action in self.actions:
                self.model[(current_state, possible_action)] = DynaQPlusTransition(
                    next_state=current_state,
                    reward=0.0,
                    terminated=False,
                    last_tried_step=0,
                )

        self.model[(current_state, action)] = DynaQPlusTransition(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            last_tried_step=self.time_step,
        )

    def _planning_reward(
        self,
        transition: DynaQPlusTransition,
    ) -> float:
        time_since_last_try = self.time_step - transition.last_tried_step
        return transition.reward + self.exploration_bonus * math.sqrt(
            time_since_last_try
        )
