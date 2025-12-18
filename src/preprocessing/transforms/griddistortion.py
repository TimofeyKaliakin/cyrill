from __future__ import annotations

from typing import Any, Dict, Tuple
import random

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class GridDistortionAugmentation(BaseAugmentation):
    """
    Аугментация локальных геометрических искажений (grid distortion)
    на базе albumentations.
    Args:
        num_steps_range - диапазон количества ячеек сетки
        distort_limit_range - диапазон искажений
    """

    name = "grid_distortion"

    def __init__(
        self,
        num_steps_range: Tuple[int, int] = (3, 6),
        distort_limit_range: Tuple[float, float] = (0.1, 0.3),
    ):
        """
        Args:
            num_steps_range - диапазон количества ячеек сетки
            distort_limit_range - диапазон искажений
        """
        self.num_steps_range = num_steps_range
        self.distort_limit_range = distort_limit_range
        self.interpolation = cv2.INTER_LINEAR
        self.normalized = True
        self.border_mode = cv2.BORDER_CONSTANT
        self.fill = 255

        # сохраняем неизменяемые параметры для логирования
        self._static_config = {
            "interpolation": self.interpolation,
            "normalized": self.normalized,
            "border_mode": self.border_mode,
            "fill_value": self.fill,
        }

    def sample_params(self) -> Dict[str, Any]:
        """
        Семплируем параметры трансформации.
        """
        num_steps = random.randint(self.num_steps_range[0],
                                   self.num_steps_range[1])
        distort_limit = random.uniform(self.distort_limit_range[0],
                                       self.distort_limit_range[1])
        return {
            "num_steps": num_steps,
            "distort_limit": distort_limit,
            **self._static_config,
        }

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        transform = A.GridDistortion(
            num_steps=params["num_steps"],
            distort_limit=params["distort_limit"],
            interpolation=params["interpolation"],
            normalized=params["normalized"],
            border_mode=params["border_mode"],
            fill=params["fill_value"],
            p=1.0,
        )
        out = transform(image=image)
        image = out["image"]

        if is_pil:
            return Image.fromarray(image)

        return image
