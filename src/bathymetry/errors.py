from __future__ import annotations


class BathymetryError(Exception):
    """Базовое исключение модуля bathymetry."""


class BathymetryNotLoadedError(BathymetryError):
    """Попытка выполнить операцию без загруженной батиметрической сетки."""


class BathymetryLoadError(BathymetryError):
    """Базовая ошибка загрузки батиметрии."""


class BathymetryConfigurationError(BathymetryLoadError):
    """Ошибка конфигурации загрузчика или источника данных."""


class BathymetryAuthenticationError(BathymetryLoadError):
    """Ошибка аутентификации/авторизации у внешнего провайдера."""


class BathymetryMissingApiKeyError(BathymetryAuthenticationError):
    """API ключ не был задан, хотя источник его требует."""

    def __init__(self, provider: str, variable_name: str | None = None) -> None:
        self.provider = provider
        self.variable_name = variable_name

        message = f"{provider} API key is required but was not provided."
        if variable_name:
            message += f" Set the `{variable_name}` variable or pass the key explicitly."
        super().__init__(message)


class BathymetryInvalidApiKeyError(BathymetryAuthenticationError):
    """API ключ передан, но не принят провайдером."""

    def __init__(self, provider: str, details: str | None = None) -> None:
        self.provider = provider
        self.details = details

        message = f"{provider} rejected the provided API key."
        if details:
            message += f" Details: {details}"
        super().__init__(message)


class BathymetryNetworkError(BathymetryLoadError):
    """Сетевая ошибка при обращении к источнику батиметрии."""


class BathymetryProviderResponseError(BathymetryLoadError):
    """Источник данных вернул неожиданный или ошибочный ответ."""

    def __init__(self, provider: str, status_code: int | None = None, details: str | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        self.details = details

        message = f"{provider} returned an invalid response"
        if status_code is not None:
            message += f" (HTTP {status_code})"
        if details:
            message += f". {details}"
        super().__init__(message)


class BathymetryDataReadError(BathymetryLoadError):
    """Ошибка чтения или декодирования полученных батиметрических данных."""
