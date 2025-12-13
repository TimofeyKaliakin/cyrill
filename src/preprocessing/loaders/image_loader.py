from __future__ import annotations

from typing import Iterator, Tuple, Optional

import numpy as np
from datasets import Dataset, DatasetDict


class HFImageLoader:
    """
    Загрузчик изображений из HuggingFace Dataset или DatasetDict.

    Args:
        dataset - HuggingFace Dataset или DatasetDict,
            с которым будем работать.
        split (default train) - имя сэмпал для загрузки (если DatasetDict).
        image_column (default image) - поле с картинкой для обработки.
            Может быть datasets.Image / Pillow.Image
        target_column (default text)

    This loader yields:
        image (np.ndarray), target, image_name
    """

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        split: str = "train",
        image_column: str = "image",
        target_column: str = "text",
        image_id_column: Optional[str] = None,
    ):
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in DatasetDict")
            self.dataset = dataset[split]
            self.split = split
        else:
            self.dataset = dataset
            self.split = "data"

        self.image_column = image_column
        self.target_column = target_column
        self.image_id_column = image_id_column

    def __len__(self) -> int:
        return len(self.dataset)

    def get_item(self, idx: int) -> Tuple[np.ndarray, str, str]:
        """
        Получить элемент по индексу. В формате numpy array.
        Возвращает кортеж (image_np, target, image_name).
        """

        item = self.dataset[idx]
        image_np = np.array(item[self.image_column])
        target = item[self.target_column]

        if self.image_id_column and self.image_id_column in item:
            image_name = str(item[self.image_id_column])
        else:
            image_name = f"{self.split}_{idx:06d}.png"

        return image_np, target, image_name

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, str]:
        return self.get_item(idx)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, str, str]]:
        for idx in range(len(self.dataset)):
            yield self.get_item(idx)
