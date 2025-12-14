from typing import Any, Dict, Tuple
import random

import cv2
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class DilationAugmentation(BaseAugmentation):
    """
    Аугментация морфологической дилатации изображения.

    Увеличивает объекты на изображении с помощью операции dilation.
    Параметры управляют размером ядра и количеством итераций.
    """

    def __init__(
        self,
        kernel_size_range: Tuple[int, int] = (3, 4),
        iterations_range: Tuple[int, int] = (1, 3),
    ):
        """
        Args:
            kernel_size_range: диапазон размеров ядра Tuple[int, int].
            iterations_range: диапазон количества итераций Tuple[int, int].
        """
        self.kernel_size_range = kernel_size_range
        self.iterations_range = iterations_range

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any]
    ) -> Image.Image | np.ndarray:
        kernel = params["kernel"]
        iterations = params["iterations"]
        return cv2.dilate(image, kernel, iterations=iterations)

    def sample_params(self) -> Dict[str, Any]:
        h = random.randint(*self.kernel_size_range)
        w = random.randint(*self.kernel_size_range)
        iterations = random.randint(*self.iterations_range)

        return {
            "kernel": np.ones((h, w), np.uint8),
            "iterations": iterations,
        }
