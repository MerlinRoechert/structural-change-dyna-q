from abc import ABC, abstractmethod

from core.action import Action
from core.grid_world import GridWorld


class Agent(ABC):
    name: str

    def __init__(self, world: GridWorld):
        self.world = world
        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.last_action: Action | None = None
        self.last_reward = 0
        self.terminated = False

    @property
    def state(self) -> tuple[int, int]:
        return self.world.current_agent_position

    def reset_episode(self) -> None:
        self.world.reset()
        self.last_action = None
        self.last_reward = 0
        self.terminated = False

    def next(self, training: bool = True) -> None:
        if self.terminated:
            return

        current_state = self.state
        action = self.choose_action(current_state, training)
        next_state, reward, terminated = self.world.step(action)

        if training:
            self.learn(
                current_state=current_state,
                action=action,
                next_state=next_state,
                reward=reward,
                terminated=terminated,
            )

        self.last_action = action
        self.last_reward = reward
        self.terminated = terminated

    @abstractmethod
    def choose_action(
        self,
        state: tuple[int, int],
        training: bool,
    ) -> Action:
        pass

    @abstractmethod
    def learn(
        self,
        current_state: tuple[int, int],
        action: Action,
        next_state: tuple[int, int],
        reward: float,
        terminated: bool,
    ) -> None:
        pass

    @abstractmethod
    def reset_learning(self) -> None:
        pass
