from enum import Enum

from PySide6.QtCore import QPoint


class VAP_Point_Type(Enum):
    NONE = 0
    PATH = 1
    TIP = 2
    BRANCH = 3


class VAP_Point():

    def __init__(self, y, x, vp_type: VAP_Point_Type):
        self.x = x
        self.y = y
        self.vp_type = vp_type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, VAP_Point):
            return self.x == other.x and self.y == other.y and self.vp_type == other.vp_type
        return False
