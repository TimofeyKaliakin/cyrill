from typing import Any, Dict, Tuple
import random

import cv2
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class ErosionAugmentation(BaseAugmentation):
    """
    Аугментация морфологической эрозии изображения.
    Параметры управляют размером ядра и количеством итераций.
    Args:
        kernal_size_range: диапазон размеров ядра Tuple[int, int].
        iterations_rnage: диапазон количества итераций Tuple[int, int].
    """
    def __init__(
        self,
        kernal_size_range: Tuple[int, int] = (3, 4),
        iterations_rnage: Tuple[int, int] = (1, 3),
    ):
        """
            Args:
                kernal_size_range: диапазон размеров ядра Tuple[int, int].
                iterations_rnage: диапазон количества итераций Tuple[int, int].
        """
        self.kernal_size_range = kernal_size_range
        self.iterations_rnage = iterations_rnage

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any]
    ) -> Image.Image | np.ndarray:

        is_pil = isinstance(image, Image.Image)

        if is_pil:
            image = np.array(image)

        kernal = params['kernal']
        iterations = params['iterations']
        image = cv2.erode(image, kernal, iterations)

        return Image.fromarray(image) if is_pil else image

    def sample_params(self) -> Dict[str, Any]:
        h = random.randint(*self.kernal_size_range)
        w = random.randint(*self.kernal_size_range)
        iterations = random.randint(*self.iterations_rnage)

        return {
            "kernal": np.ones((h, w), np.uint8),
            "iterations": iterations,
        }
