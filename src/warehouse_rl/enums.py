from enum import Enum
from pygame.math import Vector2

class RenderMode(Enum):
    NoRender = 0
    Human = 1

class Direction(Enum):
    Up = 1
    Down = 2
    Left = 3
    Right = 4

class Action(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3

NODE_SIZE = Vector2(50, 30)