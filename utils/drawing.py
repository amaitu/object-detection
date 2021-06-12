from typing import Optional, List

import cv2

from utils.utils import get_frame_height_width


def get_green_screen_colour_hex() -> tuple:
    return 4, 244, 4


def draw_crosshair(frame, midpoint: tuple, thickness: Optional[int] = 3) -> None:
    """
    midpoint: tuple (x, y)
    """

    # arbitrary number that's probably greater than the frame size
    offset = 1000

    cv2.line(
        frame,
        (midpoint[0] + offset, midpoint[1]),
        (midpoint[0] - offset, midpoint[1]),
        color=get_green_screen_colour_hex(),
        thickness=thickness,
    )

    cv2.line(
        frame,
        (midpoint[0], midpoint[1] + offset),
        (midpoint[0], midpoint[1] - offset),
        color=get_green_screen_colour_hex(),
        thickness=thickness,
    )


def draw_dot(frame, midpoint: tuple) -> None:
    cv2.circle(
        frame,
        midpoint,
        radius=0,
        color=get_green_screen_colour_hex(),
        thickness=10,
    )


def draw_annotations(frame, copy: List[str]) -> None:
    base_offset_x = 20
    base_offset_y = 50
    base_separation_distance = 30
    for text in enumerate(copy):
        cv2.putText(
            img=frame,
            text=text[1],
            org=(base_offset_x, base_offset_y + (base_separation_distance * text[0])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=get_green_screen_colour_hex(),
            thickness=2,
        )
