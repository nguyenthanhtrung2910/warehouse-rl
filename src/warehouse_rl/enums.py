from enum import Enum

from pygame.math import Vector2


class RenderMode(Enum):
    Null = 0
    Human = 1


class ObsMode(Enum):
    Flatten = 0
    ResizedWindow = 1


class Direction(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Action(Enum):
    # TODO: Force shuttle move each step can make it go to wrong direction instead of waiting other
    # If we do that, we have to add infomation about present of arounding shuttles to observations to ensure the rule:
    # one observation - one optimal action. For example, in one position we can have two optimal action:
    # 1 - go to next node if it's free, 2 - wait if next node is occupied.
    # TODO: Allow shuttle decide when to pick or drop itself. It allows us simulate parcel movement parallel with 
    # shuttle movement.
    # But if we add more action, agent needs to learn more. How we can make the balance?
    Up = 0
    Down = 1
    Left = 2
    Right = 3


NODE_SIZE = Vector2(80, 50)
STATE_SIZE = (32, 32)
