from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import random

import numpy as np
from PIL import Image

from .configs import PipelineConfig


class AugmentationPipeline:
    """
    Пайплайн аугментаций.

    Args:
        config - PipelineConfig с аугментациями и вероятностями
        seed - seed для детерминированного выбора по idx (опционально)
    """

    def __init__(self, config: PipelineConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

        self._augs = dict(self.config.augmentations)
        self._weights = dict(self.config.aug_weights)

        # Фиксируем порядок, чтобы выбор был стабильным
        self._names = list(self._augs.keys())
        self._p = [float(self._weights[name]) for name in self._names]

        # Нормализация вероятностей
        total = float(sum(self._p))
        if total <= 0:
            raise ValueError("Sum of aug_weights must be > 0")
        self._p = [p / total for p in self._p]

    def _rng(self, idx: Optional[int]) -> random.Random:
        # Если задан seed — требуем idx, чтобы выбор был стабильным на датасете HF
        if self.seed is None:
            return random

        if idx is None:
            raise ValueError("idx must be provided when seed is set")

        return random.Random(int(self.seed) + int(idx))

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        idx: Optional[int] = None,
    ) -> Tuple[Image.Image | np.ndarray, Dict[str, Any]]:
        """
        Применить аугментацию к изображению.

        Args:
            image - Изображение в формате PIL.Image или numpy.ndarray
        Returns:
            image - Аугментированное изображение
            meta - Метаданные (что применили и с какими параметрами)
        """
        rng = self._rng(idx)

        # Решаем, применяем ли аугментацию вообще
        if rng.random() > float(self.config.p_aug):
            return image, {"applied": False}

        # Выбираем одну аугментацию по вероятностям
        name = rng.choices(self._names, weights=self._p, k=1)[0]
        aug = self._augs[name]

        img_out, params = aug(image)

        meta: Dict[str, Any] = {"applied": True, "name": name}
        if self.config.return_params:
            meta["params"] = params

        return img_out, meta
