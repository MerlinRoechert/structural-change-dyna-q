from abc import ABC


class WorldObject(ABC):
    token: str = "?"
    reward: int = 0
    accessible: bool = True
    terminal: bool = False

class Wall(WorldObject):
    token: str = "#"
    reward: int = -2
    accessible: bool = False

class Candy(WorldObject):
    token: str = "c"
    reward: int = 2

class Goal(WorldObject):
    token: str = "G"
    reward: int = 10
    terminal = True

class Trap(WorldObject):
    token: str = "T"
    reward: int = -30
    terminal = True

class Slippery(WorldObject):
    token: str = "S"
    reward: int = -1
    success_probability: float = 0.6

class Empty(WorldObject):
    token: str = " "
    reward: int = -1