from typing import Any, Dict, Tuple
import random

import albumentations as A
import numpy as np
from PIL import Image

from .base import BaseAugmentation


class ElasticTransformAugmentation(BaseAugmentation):
    """
    Аугментация эластичных геометрических искажений.

    Args:
        alpha_range - диапазон силы эластического искажения
        sigma_range - диапазон сглаживания поля искажений
    """

    name = "elastic_transform"

    def __init__(
        self,
        alpha_range: Tuple[float, float] = (0.5, 2.0),
        sigma_range: Tuple[float, float] = (30.0, 60.0),
    ):
        """
        Args:
            alpha_range - диапазон силы эластического искажения
            sigma_range - диапазон сглаживания поля искажений
        """
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def sample_params(self) -> Dict[str, Any]:
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)

        return {
            "alpha": alpha,
            "sigma": sigma,
        }

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        transform = A.ElasticTransform(
            alpha=params["alpha"],
            sigma=params["sigma"],
            p=1.0,
        )

        out = transform(image=image)
        image = out["image"]

        if is_pil:
            return Image.fromarray(image)

        return image
