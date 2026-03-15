from enum import Enum

from pygame.math import Vector2


class RenderMode(Enum):
    Null = 0
    Human = 1


class ObservationMode(Enum):
    Flatten = 0
    ResizedWindow = 1
    FullWindow = 2


class Direction(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Action(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


NODE_SIZE = Vector2(100, 60)
STATE_SIZE = (96, 96)
