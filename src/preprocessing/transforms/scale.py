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
    """

    name = "scale"

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.2)):
        """
        Args:
            scale_range - Диапазон масштабирования.
                Значение выбирается случайно из этого диапазона
                при каждом применении аугментации.
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

        new_h = int(h * scale)
        new_w = int(w * scale)

        resized = cv2.resize(image_np, (new_w, new_h))

        canvas = np.ones((h, w), dtype=image_np.dtype) * 255  # white background

        # если увеличили — центрируем и обрезаем
        if scale >= 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = resized[start_y:start_y+h, start_x:start_x+w]
        else:
            # если уменьшили — паддинг
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            result = canvas

        return Image.fromarray(result) if is_pil else result
