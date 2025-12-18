from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


class BaseAugmentation(ABC):
    """
    Базовый класс для всех аугментаций.

    Совместимостим с torchvision.transforms (callable-объект)
    Совместимостим с HuggingFace datasets (map / with_transform)
    Возможно использование на инференсе (transformers)

    Каждая аугментация:
    - является вызываемым объектом
    - сама семплирует случайные параметры
    - возвращает изображение и параметры, которые были применены
    """

    name: str = "base"

    def __call__(
        self,
        image: Image.Image | np.ndarray,
    ) -> Tuple[Image.Image | np.ndarray, Dict[str, Any]]:
        """
        Применить аугментацию к изображению.

        Args:
            image - Изображение в формате PIL.Image или numpy.ndarray
        Returns
            image - Аугментированное изображение
            params - Фактически использованные параметры аугментации
        """

        params = self.sample_params()
        image = self.apply(image, params)
        return image, params

    @abstractmethod
    def sample_params(self) -> Dict[str, Any]:
        """
        Сгенерировать случайные параметры аугментации.
        Этот метод НЕ должен изменять изображение.

        Returns:
            dict - Словарь параметров аугментации
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        image: Image.Image | np.ndarray,
        params: Dict[str, Any],
    ) -> Image.Image | np.ndarray:
        """
        Применить аугментацию к изображению с заданными параметрами.

        Args:
            image - Исходное изображение
            params - Параметры аугментации
        Returns:
            image - Аугментированное изображение
        """
        raise NotImplementedError
