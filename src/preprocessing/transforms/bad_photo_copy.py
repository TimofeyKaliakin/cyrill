

from typing import Any, Dict, Tuple
import random

import numpy as np
from PIL import Image
from augraphy import BadPhotoCopy

from .base import BaseAugmentation


class BadPhotoCopyAugmentation(BaseAugmentation):
    """
    Аугментация артефактов плохой фотокопии (шум, полосы, дефекты сканирования).

    Args:
        noise_type_range - диапазон типов шума
        noise_iteration_range - диапазон количества итераций шума
        noise_size_range - диапазон размеров шумовых элементов
        noise_sparsity_range - диапазон разреженности шума
        noise_concentration_range - диапазон концентрации шума
    """

    name = "bad_photo_copy"

    def __init__(
        self,
        noise_type_range: Tuple[int, int] = (0, 3),
        noise_iteration_range: Tuple[int, int] = (1, 3),
        noise_size_range: Tuple[int, int] = (1, 3),
        noise_sparsity_range: Tuple[float, float] = (0.1, 0.5),
        noise_concentration_range: Tuple[float, float] = (0.1, 0.5),
    ):
        self.noise_type_range = noise_type_range
        self.noise_iteration_range = noise_iteration_range
        self.noise_size_range = noise_size_range
        self.noise_sparsity_range = noise_sparsity_range
        self.noise_concentration_range = noise_concentration_range

    def sample_params(self) -> Dict[str, Any]:
        noise_type = random.randint(*self.noise_type_range)
        noise_iteration = random.randint(*self.noise_iteration_range)
        noise_size = random.randint(*self.noise_size_range)
        noise_sparsity = random.uniform(*self.noise_sparsity_range)
        noise_concentration = random.uniform(*self.noise_concentration_range)

        return {
            "noise_type": noise_type,
            "noise_iteration": (noise_iteration, noise_iteration),
            "noise_size": (noise_size, noise_size),
            "noise_sparsity": (noise_sparsity, noise_sparsity),
            "noise_concentration": (noise_concentration, noise_concentration),
        }

    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image = np.array(image)

        transform = BadPhotoCopy(
            noise_type=params["noise_type"],
            noise_side="random",
            noise_iteration=params["noise_iteration"],
            noise_size=params["noise_size"],
            noise_value=(32, 128),
            noise_sparsity=params["noise_sparsity"],
            noise_concentration=params["noise_concentration"],
            blur_noise=-1,
            wave_pattern=-1,
            edge_effect=-1,
            p=1,
        )

        image = transform(image)

        if is_pil:
            return Image.fromarray(image)

        return image
