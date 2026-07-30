from collections.abc import Iterator

from core.world_object import Empty, WorldObject


class WorldMap:
    MAX_STATES = 2

    def __init__(
        self,
        states: dict[int, list[list[WorldObject]]],
        initial_state: int = 1,
    ):
        self._validate_states(states, initial_state)
        self._states = {
            state_id: [
                [type(cell) for cell in row]
                for row in cells
            ]
            for state_id, cells in states.items()
        }
        self.current_state = initial_state
        self._cells = self._create_cells(initial_state)

    @property
    def state_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._states))

    def set_state(self, state_id: int) -> None:
        if state_id not in self._states:
            raise ValueError(
                f"Unknown map state '{state_id}'. Available: {self.state_ids}"
            )

        target_state = self._states[state_id]

        for row_index, target_row in enumerate(target_state):
            for column_index, target_cell_type in enumerate(target_row):
                current_cell = self._cells[row_index][column_index]

                if type(current_cell) is target_cell_type:
                    continue

                self._cells[row_index][column_index] = target_cell_type()

        self.current_state = state_id

    def set_cell(
        self,
        position: tuple[int, int],
        cell: WorldObject,
    ) -> None:
        row, column = position
        self._cells[row][column] = cell
        self._states[self.current_state][row][column] = type(cell)

    def insert_column(self, column_index: int) -> None:
        for row in self._cells:
            row.insert(column_index, Empty())

        for state in self._states.values():
            for row in state:
                row.insert(column_index, Empty)

    def remove_column(self, column_index: int) -> None:
        for row in self._cells:
            row.pop(column_index)

        for state in self._states.values():
            for row in state:
                row.pop(column_index)

    def insert_row(self, row_index: int) -> None:
        width = len(self._cells[0])
        self._cells.insert(row_index, [Empty() for _ in range(width)])

        for state in self._states.values():
            state.insert(row_index, [Empty for _ in range(width)])

    def remove_row(self, row_index: int) -> None:
        self._cells.pop(row_index)

        for state in self._states.values():
            state.pop(row_index)

    def __getitem__(self, row_index: int) -> list[WorldObject]:
        return self._cells[row_index]

    def __iter__(self) -> Iterator[list[WorldObject]]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def _create_cells(self, state_id: int) -> list[list[WorldObject]]:
        return [
            [cell_type() for cell_type in row]
            for row in self._states[state_id]
        ]

    def _validate_states(
        self,
        states: dict[int, list[list[WorldObject]]],
        initial_state: int,
    ) -> None:
        if not states:
            raise ValueError("WorldMap requires at least one state")

        if len(states) > self.MAX_STATES:
            raise ValueError("WorldMap supports at most two states")

        if 1 not in states:
            raise ValueError("WorldMap requires state 1")

        if any(state_id not in (1, 2) for state_id in states):
            raise ValueError("WorldMap state ids must be 1 or 2")

        if initial_state not in states:
            raise ValueError("initial_state must exist in states")

        expected_size: tuple[int, int] | None = None

        for cells in states.values():
            if not cells or not cells[0]:
                raise ValueError("map states must not be empty")

            width = len(cells[0])
            if any(len(row) != width for row in cells):
                raise ValueError("map states must be rectangular")

            size = (len(cells), width)
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("all map states must have the same size")
