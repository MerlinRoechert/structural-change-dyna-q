import random
from abc import ABC, abstractmethod

from core.world_map import WorldMap


class WorldStateBehavior(ABC):
    log_state_changes: bool

    @abstractmethod
    def update(self, world_map: WorldMap, current_step: int) -> None:
        pass


class StaticWorldStateBehavior(WorldStateBehavior):
    log_state_changes = False

    def update(self, world_map: WorldMap, current_step: int) -> None:
        pass


class ScheduledWorldStateBehavior(WorldStateBehavior):
    log_state_changes = True

    def __init__(
        self,
        change_step: int,
        change_duration: int | None,
    ):
        self.change_step = change_step
        self.change_duration = change_duration

    def update(self, world_map: WorldMap, current_step: int) -> None:
        if current_step == self.change_step:
            world_map.set_state(2)

        if (
            self.change_duration is not None
            and current_step == self.change_step + self.change_duration
        ):
            world_map.set_state(1)


class TransientNoiseWorldStateBehavior(WorldStateBehavior):
    log_state_changes = False
    NOISE_STATE_IDS = (2, 3)
    NOISE_PROBABILITY = 0.2

    def __init__(
        self,
        noise_start_step: int,
        noise_duration: int | None,
        seed: int,
    ):
        self.noise_start_step = noise_start_step
        self.noise_duration = noise_duration
        self.random = random.Random(seed)

    def update(self, world_map: WorldMap, current_step: int) -> None:
        if not self._noise_is_active(current_step):
            if world_map.current_state != 1:
                world_map.set_state(1)
            return

        if world_map.current_state != 1:
            world_map.set_state(1)
            return

        if self.random.random() < self.NOISE_PROBABILITY:
            world_map.set_state(self.random.choice(self.NOISE_STATE_IDS))

    def _noise_is_active(self, current_step: int) -> bool:
        if current_step < self.noise_start_step:
            return False

        if self.noise_duration is None:
            return True

        return current_step < self.noise_start_step + self.noise_duration
