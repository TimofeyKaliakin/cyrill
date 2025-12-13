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

        # Приводим изображение к numpy
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_np = np.array(image)
        else:
            image_np = image

        height, width = image_np.shape[:2]

        # Центр изображения, относительно которого масштабируем
        center_x = width / 2
        center_y = height / 2

        # Матрица масштабирования вокруг центра
        M = np.array(
            [
                [scale, 0, (1 - scale) * center_x],
                [0, scale, (1 - scale) * center_y],
            ],
            dtype=np.float32,
        )

        # Применяем affine-преобразование
        scaled = cv2.warpAffine(
            image_np,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Возвращаем тип изображения в исходном формате
        if is_pil:
            return Image.fromarray(scaled)

        return scaled
