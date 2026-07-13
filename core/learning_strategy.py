from abc import ABC
from enum import Enum

from core.action import Action


class LearningStrategyName(Enum):
    Q_LEARNING = "q-learning"

class LearningStrategy(ABC):
    name: str = ""
    def update(self,
        q_table: dict[tuple[int, int], dict[Action, float]],
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: int,
        terminated: bool,
        discount_factor: float,
        learning_rate: float
    ):
        raise NotImplementedError()
    
class QLearning(LearningStrategy):
    name: str = LearningStrategyName.Q_LEARNING
    def update(self,
        q_table: dict[tuple[int, int], dict[Action, float]],
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: int,
        terminated: bool,
        discount_factor: float,
        learning_rate: float
    ):
        old_q = q_table[current_state][action]

        if terminated:
            best_next_q = 0.0
        else:
            best_next_q = max(q_table[next_state].values())

        return old_q + learning_rate * (
            reward + discount_factor * best_next_q - old_q
        )
