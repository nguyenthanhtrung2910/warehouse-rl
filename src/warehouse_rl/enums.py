from enum import Enum

class RenderMode(Enum):
    NoRender = 1
    Human = 2

class Direction(Enum):
    Up = 1
    Down = 2
    Left = 3
    Right = 4

class Action(Enum):
    # NONE = 0
    Up = 1
    Down = 2
    Left = 3
    Right = 4