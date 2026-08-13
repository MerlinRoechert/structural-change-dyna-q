import random

from core.action import ACTION_DELTAS, Action
from core.world_map import WorldMap
from core.world_object import Candy, Empty, Goal, Slippery, Trap, Wall, WorldObject
from core.world_state_behavior import WorldStateBehavior

class GridWorld:
    def __init__(
        self,
        start_position: tuple[int, int],
        state_behavior: WorldStateBehavior,
        seed: int,
        world_map: WorldMap | None = None,
    ):
        self.start_position: tuple[int, int] = start_position
        self.current_agent_position: tuple[int, int] = start_position
        self.state_behavior = state_behavior
        self.random = random.Random(seed)
        if world_map is None:
            world_map = self._create_default_map()

        self.map = world_map

        self._validate_map()
        self.height = len(self.map)
        self.width = len(self.map[0])

    def update_state(self, current_step: int) -> bool:
        previous_state = self.map.current_state
        self.state_behavior.update(self.map, current_step)
        return previous_state != self.map.current_state

    def _create_default_map(self) -> WorldMap:
        return WorldMap(
            states={
                1: [
                    [Empty(), Empty(), Empty(), Empty(), Wall()],
                    [Empty(), Wall(), Wall(), Empty(), Empty()],
                    [Empty(), Empty(), Trap(), Empty(), Wall()],
                    [Trap(), Empty(), Empty(), Empty(), Empty()],
                    [Empty(), Empty(), Wall(), Goal(), Wall()],
                ]
            }
        )

    def _validate_map(self):
        if not self.map or not self.map[0]:
            raise ValueError("world_map must not be empty")

        width = len(self.map[0])
        if any(len(row) != width for row in self.map):
            raise ValueError("world_map must be rectangular")

        start_row, start_col = self.start_position
        if not 0 <= start_row < len(self.map):
            raise ValueError("start row is outside world_map")

        if not 0 <= start_col < width:
            raise ValueError("start column is outside world_map")

        if not self.map[start_row][start_col].accessible:
            raise ValueError("start position must be accessible")

    def print_map(self):
        for r, row in enumerate(self.map):
            for c, col in enumerate(row):
                if (r, c) == self.current_agent_position:
                    print("A", end=" ")
                else:
                    print(col.token, end=" ")
            print()

    def reset(self) -> tuple[int, int]:
        self.current_agent_position = self.start_position
        self.map.set_state(self.map.current_state)
        return self.current_agent_position

    def get_cell(self, position: tuple[int , int]) -> WorldObject:
        row, col = position
        if row < 0 or row >= self.height:
            raise Exception("row not on map")

        if col < 0 or col >= self.width:
            raise Exception("col not on map")
        
        return self.map[row][col]
    
    def step(self, action: Action) -> tuple[tuple[int, int], int, bool]:
        row, col = self.current_agent_position

        current_cell = self.get_cell(self.current_agent_position)

        if isinstance(current_cell, Slippery):
            if self.random.random() > current_cell.success_probability:
                possible_actions = [
                    possible_action
                    for possible_action in ACTION_DELTAS.keys()
                    if possible_action != action
                ]
                action = self.random.choice(possible_actions)

        row_delta, col_delta = ACTION_DELTAS[action]
        new_position = (row + row_delta, col + col_delta)

        try:
            cell = self.get_cell(new_position)
        except Exception:
            return self.current_agent_position, -1, False

        if not cell.accessible:
            return self.current_agent_position, cell.reward, False

        self.current_agent_position = new_position

        reward = cell.reward
        terminal = cell.terminal

        if isinstance(cell, Candy):
            new_row, new_col = new_position
            self.map[new_row][new_col] = Empty()

        return new_position, reward, terminal
    
    def insert_column(self, col_index: int):
        self.map.insert_column(col_index)

        self.width = len(self.map[0])


    def remove_column(self, col_index: int):
        if self.width <= 1:
            return

        self.map.remove_column(col_index)

        self.width = len(self.map[0])

        self.start_position = self._clamp_position(self.start_position)
        self.current_agent_position = self._clamp_position(self.current_agent_position)


    def insert_row(self, row_index: int):
        self.map.insert_row(row_index)

        self.height = len(self.map)


    def remove_row(self, row_index: int):
        if self.height <= 1:
            return

        self.map.remove_row(row_index)

        self.height = len(self.map)

        self.start_position = self._clamp_position(self.start_position)
        self.current_agent_position = self._clamp_position(self.current_agent_position)


    def _clamp_position(self, position: tuple[int, int]) -> tuple[int, int]:
        row, col = position

        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))

        return row, col
    
    def set_cell(self, position: tuple[int, int], cell: WorldObject):
        self.map.set_cell(position, cell)

    def set_start_position(self, position: tuple[int, int]):
        self.start_position = position
        self.current_agent_position = position
