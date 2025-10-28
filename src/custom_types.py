from typing import Literal, TypedDict


class BBox(TypedDict):
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float


class HitBox(BBox):
    class_name: Literal["TitleBlock", "View", "Note"]
