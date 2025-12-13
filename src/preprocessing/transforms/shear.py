from typing import Any, Dict, Optional
import random
import math

import cv2
import numpy as np
from PIL import Image

from .base import BaseAugmentation
from .scale import ScaleAugmentation

class ShearAugmentation(BaseAugmentation):
    """
    Аугментация shear (сдвиг) по оси X и/или Y.
    """
    def __init__(
        self,
        shear_x_range: Optional[tuple[float, float]] = None,
        shear_y_range: Optional[tuple[float, float]] = None,
    ):
        """
        Инициализация параметров shear-аугментации.

        Args:
            shear_x_range - диапазон углов (в градусах) для shear по оси X
            shear_y_range - диапазон углов (в градусах) для shear по оси Y
        """
        self.shear_x_range = shear_x_range
        self.shear_y_range = shear_y_range
        self.scaler = ScaleAugmentation(scale_range=(0.7, 0.8))

        if shear_x_range is None and shear_y_range is None:
            raise ValueError("At least one of shear_x_range or shear_y_range must be specified") # noqa

    def sample_params(self) -> Dict[str, Any]:
        """
        Сгенерировать параметры сдвига по осям.

        Returns:
            params - параметры shear-аугментации
        """
        params: Dict[str, Any] = {}

        if self.shear_x_range is not None:
            phi_x = random.uniform(*self.shear_x_range)
            kx = math.tan(math.radians(phi_x))  # shear по X
        else:
            phi_x = 0.0
            kx = 0.0

        if self.shear_y_range is not None:
            phi_y = random.uniform(*self.shear_y_range)
            ky = math.tan(math.radians(phi_y))  # shear по Y
        else:
            phi_y = 0.0
            ky = 0.0

        params["phi_x"] = phi_x
        params["phi_y"] = phi_y
        params["kx"] = kx
        params["ky"] = ky

        return params

    def apply(self, image: Any, params: Dict[str, Any]) -> Any:
        """
        Применить аугментацию к изображению.

        Args:
            image - Изображение в формате PIL.Image или numpy.ndarray
        Returns:
            image - Аугментированное изображение
            params - Фактически использованные параметры аугментации
        """
        kx = params["kx"]
        ky = params["ky"]

        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        h, w = image.shape[:2]

        # Базовая shear-матрица
        A = np.array([[1.0, kx], [ky, 1.0]], dtype=np.float32)

        # Сдвиг для shear относительно центра изображения
        cx, cy = w / 2, h / 2
        tx = cx - (A[0, 0] * cx + A[0, 1] * cy)
        ty = cy - (A[1, 0] * cx + A[1, 1] * cy)

        M = np.array([[A[0, 0], A[0, 1], tx],
                      [A[1, 0], A[1, 1], ty]], dtype=np.float32)

        # Добавляем уменьшение изображения, чтобы избежать выхода за границы
        scaled_img, _ = self.scaler(image)

        sheared = cv2.warpAffine(
            scaled_img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        if is_pil:
            sheared = Image.fromarray(sheared)

        return sheared
