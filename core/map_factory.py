from enum import Enum

from core.grid_world import GridWorld
from core.world_map import WorldMap
from core.world_object import Empty, Goal, Slippery, Trap, Wall, WorldObject
from core.world_state_behavior import (
    ScheduledWorldStateBehavior,
    StaticWorldStateBehavior,
    TransientNoiseWorldStateBehavior,
    WorldStateBehavior,
)


class WorldId(str, Enum):
    TWO_ROUTES = "two-routes"
    SHORTCUT = "shortcut"
    MIXED_STABILITY = "mixed-stability"
    TRANSIENT_NOISE = "transient-noise"


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
        WorldId.TRANSIENT_NOISE: {
            1: (
                "###########",
                "#A.......G#",
                "#.#######.#",
                "#.........#",
                "###########",
            ),
            2: (
                "###########",
                "#A.T.....G#",
                "#.#######.#",
                "#.........#",
                "###########",
            ),
            3: (
                "###########",
                "#A.....T.G#",
                "#.#######.#",
                "#.........#",
                "###########",
            ),
        },
    }

    _CELL_TYPES = {
        ".": Empty,
        "#": Wall,
        "G": Goal,
        "T": Trap,
        "~": Slippery,
    }

    @classmethod
    def create(
        cls,
        world_id: WorldId | str,
        change_step: int | None = None,
        change_duration: int | None = None,
        seed: int = 0,
    ) -> GridWorld:
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

        state_behavior = cls._create_state_behavior(
            world_id=normalized_world_id,
            change_step=change_step,
            change_duration=change_duration,
            seed=seed,
        )

        return GridWorld(
            start_position=start_position,
            state_behavior=state_behavior,
            seed=seed,
            world_map=WorldMap(state_maps),
        )

    @classmethod
    def _create_state_behavior(
        cls,
        world_id: WorldId,
        change_step: int | None,
        change_duration: int | None,
        seed: int,
    ) -> WorldStateBehavior:
        if change_step is None:
            return StaticWorldStateBehavior()

        if world_id is WorldId.TRANSIENT_NOISE:
            return TransientNoiseWorldStateBehavior(
                noise_start_step=change_step,
                noise_duration=change_duration,
                seed=seed,
            )

        return ScheduledWorldStateBehavior(
            change_step=change_step,
            change_duration=change_duration,
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
