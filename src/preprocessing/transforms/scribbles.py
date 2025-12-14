

from typing import Any, Dict, Tuple
import random

import numpy as np
from PIL import Image
from augraphy.augmentations.scribbles import Scribbles

from .base import BaseAugmentation


class ScribblesAugmentation(BaseAugmentation):
    """
    Аугментация случайных пометок, линий и надписей (scribbles).

    Args:
        size_range - диапазон размеров пометок
        count_range - диапазон количества пометок
        thickness_range - диапазон толщины линий
        brightness_values - возможные изменения яркости
        rotation_range - диапазон углов поворота текста
    """

    name = "scribbles"

    def __init__(
        self,
        size_range: Tuple[int, int] = (400, 600),
        count_range: Tuple[int, int] = (1, 6),
        thickness_range: Tuple[int, int] = (1, 3),
        brightness_values: Tuple[int, ...] = (32, 64, 128),
        rotation_range: Tuple[int, int] = (0, 360),
    ):
        self.size_range = size_range
        self.count_range = count_range
        self.thickness_range = thickness_range
        self.brightness_values = brightness_values
        self.rotation_range = rotation_range

    def sample_params(self) -> Dict[str, Any]:
        size = random.randint(*self.size_range)
        count = random.randint(*self.count_range)
        thickness = random.randint(*self.thickness_range)
        brightness = random.choice(self.brightness_values)
        rotation = random.randint(*self.rotation_range)

        return {
            "size": (size, size),
            "count": (count, count),
            "thickness": (thickness, thickness),
            "brightness": brightness,
            "rotation": (rotation, rotation),
        }

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        transform = Scribbles(
            scribbles_type="lines",
            scribbles_ink="random",
            scribbles_location="random",
            scribbles_size_range=params["size"],
            scribbles_count_range=params["count"],
            scribbles_thickness_range=params["thickness"],
            scribbles_brightness_change=[params["brightness"]],
            scribbles_color="random",
            scribbles_text=None,
            scribbles_text_font="random",
            scribbles_text_rotate_range=params["rotation"],
            p=1,
        )

        image = transform(image)

        return Image.fromarray(image) if is_pil else image
