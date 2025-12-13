from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PipelineConfig:
    """
    Конфигурация пайплайна аугментаций.

    Args:
        p_aug:
            Общая вероятность применения аугментации к изображению.

        augmentations:
            Словарь аугментаций.
            Ключ   — имя аугментации (augmentation.name)
            Значение — объект аугментации с заданными параметрами.

        aug_weights:
            Словарь аугментаций с вероятностью выбора аугментаций.
            Ключ   — имя аугментации (augmentation.name)
            Значение — вероятность применения аугментации.
            Если не заданы, считаются одинаковыми для всех аугментаций.

        return_params:
            Нужно ли возвращать параметры аугментации в выходном примере.
    """

    p_aug: float = 0.5
    augmentations: Dict[str, Any] = field(default_factory=dict)
    aug_weights: Dict[str, float] = field(default_factory=dict)
    return_params: bool = True

    def __post_init__(self) -> None:
        # Проверка p_aug
        if not 0.0 <= self.p_aug <= 1.0:
            raise ValueError("p_aug must be in [0, 1]")

        # Проверка аугментаций
        if not self.augmentations:
            raise ValueError("augmentations must not be empty")

        # Если веса не заданы — делаем равные
        if not self.aug_weights:
            self.aug_weights = {name: 1.0 / len(self.augmentations.keys()) for name in self.augmentations}

        # Проверка совпадения ключей
        if set(self.augmentations.keys()) != set(self.aug_weights.keys()):
            raise ValueError(
                "Keys of augmentations and aug_weights must match exactly"
            )

        # Проверка весов
        total_weight = 0.0
        for name, weight in self.aug_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for augmentation '{name}' must be >= 0"
                )
            total_weight += weight

        if not 0.98 <= total_weight <= 1.01:
            raise ValueError("Sum of aug_weights must be == 1")
