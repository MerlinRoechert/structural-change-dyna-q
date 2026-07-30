from enum import Enum

from core.grid_world import GridWorld
from core.world_map import WorldMap
from core.world_object import Empty, Goal, Slippery, Wall, WorldObject


class WorldId(str, Enum):
    TWO_ROUTES = "two-routes"
    SHORTCUT = "shortcut"
    MIXED_STABILITY = "mixed-stability"


class MapFactory:
    _STATE_LAYOUTS = {
        WorldId.TWO_ROUTES: {
            1: (
                "###########",
                "#A.......G#",
                "#.#######.#",
                "#.........#",
                "#.#######.#",
                "#.........#",
                "###########",
            ),
            2: (
                "###########",
                "#A...#...G#",
                "#.#######.#",
                "#.........#",
                "#.#######.#",
                "#.........#",
                "###########",
            ),
        },
        WorldId.SHORTCUT: {
            1: (
                "###########",
                "#A...#...G#",
                "#....#....#",
                "#....#....#",
                "#.........#",
                "###########",
            ),
            2: (
                "###########",
                "#A...#...G#",
                "#.........#",
                "#....#....#",
                "#.........#",
                "###########",
            ),
        },
        WorldId.MIXED_STABILITY: {
            1: (
                "###########",
                "#A..~~~...#",
                "#.#.....#.#",
                "#.#.###.#.#",
                "#...~~~...#",
                "#.......#G#",
                "###########",
            ),
            2: (
                "###########",
                "#A..~~~#..#",
                "#.#.....#.#",
                "#.#.###.#.#",
                "#...~~~...#",
                "#.......#G#",
                "###########",
            ),
        },
    }

    _CELL_TYPES = {
        ".": Empty,
        "#": Wall,
        "G": Goal,
        "~": Slippery,
    }

    @classmethod
    def create(cls, world_id: WorldId | str) -> GridWorld:
        try:
            normalized_world_id = WorldId(world_id)
        except ValueError as error:
            available_ids = ", ".join(item.value for item in WorldId)
            raise ValueError(
                f"Unknown world_id '{world_id}'. Available: {available_ids}"
            ) from error

        state_maps: dict[int, list[list[WorldObject]]] = {}
        start_position: tuple[int, int] | None = None

        for state_id, layout in cls._STATE_LAYOUTS[normalized_world_id].items():
            state_map, state_start_position = cls._parse_layout(layout)

            if start_position is None:
                start_position = state_start_position
            elif state_start_position != start_position:
                raise ValueError("all map states must use the same start")

            state_maps[state_id] = state_map

        return GridWorld(
            start_position=start_position,
            world_map=WorldMap(state_maps),
        )

    @classmethod
    def _parse_layout(
        cls,
        layout: tuple[str, ...],
    ) -> tuple[list[list[WorldObject]], tuple[int, int]]:
        world_map: list[list[WorldObject]] = []
        start_position: tuple[int, int] | None = None
        goal_count = 0

        for row_index, row in enumerate(layout):
            world_row: list[WorldObject] = []

            for col_index, token in enumerate(row):
                if token == "A":
                    if start_position is not None:
                        raise ValueError("layout must contain exactly one start")

                    start_position = (row_index, col_index)
                    world_row.append(Empty())
                    continue

                cell_type = cls._CELL_TYPES.get(token)
                if cell_type is None:
                    raise ValueError(f"Unknown map token '{token}'")

                if cell_type is Goal:
                    goal_count += 1

                world_row.append(cell_type())

            world_map.append(world_row)

        if start_position is None:
            raise ValueError("layout must contain exactly one start")

        if goal_count != 1:
            raise ValueError("layout must contain exactly one goal")

        return world_map, start_position
