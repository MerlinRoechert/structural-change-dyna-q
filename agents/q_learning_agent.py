import random

from agents.agent import Agent
from core.action import Action
from core.grid_world import GridWorld


class QLearningAgent(Agent):
    name = "q-learning"

    def __init__(
        self,
        world: GridWorld,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
    ):
        super().__init__(world)
        self.q_table: dict[tuple[int, int], dict[Action, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(
        self,
        state: tuple[int, int],
        training: bool,
    ) -> Action:
        self.init_state(state)

        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        return max(self.q_table[state], key=self.q_table[state].get)

    def learn(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        self.init_state(current_state)
        self.init_state(next_state)

        old_q = self.q_table[current_state][action]
        best_next_q = 0.0 if terminated else max(
            self.q_table[next_state].values()
        )

        self.q_table[current_state][action] = old_q + self.learning_rate * (
            reward + self.discount_factor * best_next_q - old_q
        )

    def init_state(self, state: tuple[int, int]) -> None:
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.actions
            }

    def reset_learning(self) -> None:
        self.q_table.clear()

    def print_q_table(self) -> None:
        for state, actions in self.q_table.items():
            action_values = ", ".join(
                f"{action.name}: {value:.2f}"
                for action, value in actions.items()
            )
            print(f"{state} -> {action_values}")
