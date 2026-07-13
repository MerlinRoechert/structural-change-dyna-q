import random

from core.action import Action
from core.learning_strategy import LearningStrategy, QLearning

class Agent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.2, learning_strategy: LearningStrategy = QLearning()):
        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.q_table: dict[tuple[int, int], dict[Action, float]] = {}
        self.learning_rate: float = learning_rate
        self.discount_factor: float = discount_factor
        self.epsilon: float = epsilon
        self.learning_strategy: LearningStrategy = learning_strategy

    def print_q_table(self):
        for state, actions in self.q_table.items():
            action_str = ", ".join(
                f"{a.name}: {v:.2f}" for a, v in actions.items()
            )
            print(f"{state} -> {action_str}")

    def init_state(self, state: tuple[int, int]):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

    def choose_action(self, state, epsilon_override=None):
        epsilon = self.epsilon if epsilon_override is None else epsilon_override

        self.init_state(state)

        if random.random() < epsilon:
            return random.choice(self.actions)

        return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, current_state: tuple[int, int], action: Action, next_state: tuple[int, int], reward: int, terminated: bool):
        self.init_state(current_state)
        self.init_state(next_state)
        
        if self.learning_strategy is not None:
            self.q_table[current_state][action] = self.learning_strategy.update(
                q_table=self.q_table,
                current_state=current_state,
                action=action,
                next_state=next_state,
                reward=reward,
                terminated=terminated,
                discount_factor=self.discount_factor,
                learning_rate=self.learning_rate
            )

    