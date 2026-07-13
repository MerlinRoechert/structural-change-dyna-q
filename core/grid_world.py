import random
from typing import Tuple

from core.action import ACTION_DELTAS, Action
from core.world_object import Candy, Empty, Goal, Slippery, Trap, Wall, WorldObject

class GridWorld:
    def __init__(self, start_position: tuple[int, int]):
        self.start_position: tuple[int, int] = start_position
        self.current_agent_position: tuple[int, int] = start_position
        self.candys: list[tuple[int, int]] = []
        self.map: list[list[WorldObject]] = [
            [Empty(), Empty(), Empty(), Empty(), Wall()],
            [Empty(), Wall(), Wall(), Empty(), Empty()],
            [Empty(), Empty(), Trap(), Empty(), Wall()],
            [Trap(), Empty(), Empty(), Empty(), Empty()],
            [Empty(), Empty(), Wall(), Goal(), Wall()],
        ]
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.collect_candys()

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
        self.restore_candys()
        return self.current_agent_position
    
    def restore_candys(self):
        for position in self.candys:
            self.map[position[0]][position[1]] = Candy()

    def collect_candys(self):
        self.candys = []

        for r, row in enumerate(self.map):
            for c, cell in enumerate(row):
                if isinstance(cell, Candy):
                    self.candys.append((r, c))

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
            if random.random() > current_cell.success_probability:
                possible_actions = [
                    possible_action
                    for possible_action in ACTION_DELTAS.keys()
                    if possible_action != action
                ]
                action = random.choice(possible_actions)

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
        for row in self.map:
            row.insert(col_index, Empty())

        self.width = len(self.map[0])
        self.collect_candys()


    def remove_column(self, col_index: int):
        if self.width <= 1:
            return

        for row in self.map:
            row.pop(col_index)

        self.width = len(self.map[0])

        self.start_position = self._clamp_position(self.start_position)
        self.current_agent_position = self._clamp_position(self.current_agent_position)
        self.collect_candys()


    def insert_row(self, row_index: int):
        new_row = [Empty() for _ in range(self.width)]
        self.map.insert(row_index, new_row)

        self.height = len(self.map)
        self.collect_candys()


    def remove_row(self, row_index: int):
        if self.height <= 1:
            return

        self.map.pop(row_index)

        self.height = len(self.map)

        self.start_position = self._clamp_position(self.start_position)
        self.current_agent_position = self._clamp_position(self.current_agent_position)
        self.collect_candys()


    def _clamp_position(self, position: tuple[int, int]) -> tuple[int, int]:
        row, col = position

        row = max(0, min(row, self.height - 1))
        col = max(0, min(col, self.width - 1))

        return row, col
    
    def set_cell(self, position: tuple[int, int], cell: WorldObject):
        row, col = position

        old_cell = self.get_cell(position)

        if isinstance(old_cell, Candy) and position in self.candys:
            self.candys.remove(position)

        self.map[row][col] = cell

        if isinstance(cell, Candy) and position not in self.candys:
            self.candys.append(position)

    def set_start_position(self, position: tuple[int, int]):
        self.start_position = position
        self.current_agent_position = position