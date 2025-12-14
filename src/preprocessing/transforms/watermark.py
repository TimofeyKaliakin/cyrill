

from typing import Any, Dict, Tuple
import random

import numpy as np
from PIL import Image
from augraphy.augmentations.watermark import WaterMark

from .base import BaseAugmentation


class WaterMarkAugmentation(BaseAugmentation):
    """
    Аугментация добавления водяных знаков.

    Args:
        words - набор слов для водяных знаков
        font_size_range - диапазон размеров шрифта
        font_thickness_range - диапазон толщины шрифта
        rotation_range - диапазон углов поворота
    """

    name = "watermark"

    def __init__(
        self,
        words: Tuple[str, ...],
        font_size_range: Tuple[int, int] = (10, 20),
        font_thickness_range: Tuple[int, int] = (1, 3),
        rotation_range: Tuple[int, int] = (0, 360),
    ):
        if not words:
            raise ValueError("words must contain at least one watermark string") # noqa

        self.words = words
        self.font_size_range = font_size_range
        self.font_thickness_range = font_thickness_range
        self.rotation_range = rotation_range

    def sample_params(self) -> Dict[str, Any]:
        word = random.choice(self.words)
        font_size = random.randint(*self.font_size_range)
        font_thickness = random.randint(*self.font_thickness_range)
        rotation = random.randint(*self.rotation_range)

        return {
            "word": word,
            "font_size": (font_size, font_size),
            "font_thickness": (font_thickness, font_thickness),
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

        transform = WaterMark(
            watermark_word=params["word"],
            watermark_font_size=params["font_size"],
            watermark_font_thickness=params["font_thickness"],
            watermark_rotation=params["rotation"],
            watermark_location="random",
            watermark_color="random",
            watermark_method="darken",
            p=1,
        )

        image = transform(image)

        return Image.fromarray(image) if is_pil else image
