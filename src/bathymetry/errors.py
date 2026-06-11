from __future__ import annotations


class BathymetryError(Exception):
    """Базовая ошибка модуля батиметрии."""


class BathymetryLoadError(BathymetryError):
    """Ошибка загрузки батиметрических данных."""


class BathymetryNotLoadedError(BathymetryError):
    """Сетка батиметрии ещё не загружена."""
