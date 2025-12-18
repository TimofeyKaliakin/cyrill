from typing import Any, Dict
import random

import cv2
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class ScaleAugmentation(BaseAugmentation):
    """
    Аугментация масштабирования (увеличение / уменьшение изображения).

    Масштабирование выполняется относительно центра изображения,
    что позволяет корректно работать с изображениями любой размерности.

    Args:
        scale_range - Диапазон масштабирования.
            Значение выбирается случайно из этого диапазона
            при каждом применении аугментации.
            Нет, смысла передавать что-то больше 1, тк вернется оригинал.
    """

    name = "scale"

    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.0)):
        """
        Args:
            scale_range - Диапазон масштабирования.
                Значение выбирается случайно из этого диапазона
                при каждом применении аугментации.
                Нет, смысла передавать что-то больше 1, тк вернется оригинал.
        """
        self.scale_range = scale_range

    def sample_params(self,) -> Dict[str, Any]:
        """
        Случайно сэмплировать коэффициент масштабирования.
        """
        scale = random.uniform(*self.scale_range)
        return {"scale": scale}

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        """
        Применить масштабирование к изображению.
        """

        scale = params["scale"]

        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_np = np.array(image)
        else:
            image_np = image

        h, w = image_np.shape[:2]

        if scale >= 1.0:
            return image if is_pil else image_np

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(
            image_np,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        if image_np.ndim == 2:  # grayscale
            canvas = np.full((h, w), 255, dtype=image_np.dtype)
        else:  # RGB
            canvas = np.full((h, w, image_np.shape[2]), 255, dtype=image_np.dtype)

        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        return Image.fromarray(canvas) if is_pil else canvas
