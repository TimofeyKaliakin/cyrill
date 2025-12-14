from typing import Any, Dict, Tuple
import random

import albumentations as A
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class MotionBlurAugmentation(BaseAugmentation):
    """
    Аугментация эффекта motion blur (смазывание движения).
    """

    name = "motion_blur"

    def __init__(
        self,
        blur_limit_range: Tuple[int, int] = (3, 7),
        angle_range: Tuple[float, float] = (0.0, 360.0),
        direction_range: Tuple[float, float] = (-1.0, 1.0),
        allow_shifted: bool = True,
    ):
        """
        Args:
            blur_limit_range - диапазон размера ядра размытия
            angle_range - диапазон углов размытия (в градусах)
            direction_range - диапазон направленности размытия
            allow_shifted - разрешать ли смещение ядра
        """
        self.blur_limit_range = blur_limit_range
        self.angle_range = angle_range
        self.direction_range = direction_range
        self.allow_shifted = allow_shifted

    def sample_params(self) -> Dict[str, Any]:
        blur_limit = random.randint(*self.blur_limit_range)
        angle = random.uniform(*self.angle_range)
        direction = random.uniform(*self.direction_range)

        return {
            "blur_limit": blur_limit,
            "angle": angle,
            "direction": direction,
            "allow_shifted": self.allow_shifted,
        }

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        transform = A.MotionBlur(
            blur_limit=params["blur_limit"],
            angle_range=(params["angle"], params["angle"]),
            direction_range=(params["direction"], params["direction"]),
            allow_shifted=params["allow_shifted"],
            p=1.0,
        )

        out = transform(image=image)
        image = out["image"]

        if is_pil:
            return Image.fromarray(image)

        return image
