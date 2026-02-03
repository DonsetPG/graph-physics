import enum


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    ANEURYSM = 7
    SIZE = 9


GLOBAL_ATTENTION_NODE = NodeType.WALL_BOUNDARY
