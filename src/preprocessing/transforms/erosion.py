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

        kernal = params['kernal']
        iterations = params['iterations']
        return cv2.erode(image, kernal, iterations)

    def sample_params(self) -> Dict[str, Any]:
        h = random.uniform(*self.kernal_size_range)
        w = random.uniform(*self.kernal_size_range)

        return {'kernal': np.ones((h, w), np.uint8),
                'iterations': random.uniform(*self.iterations_rnage)}
