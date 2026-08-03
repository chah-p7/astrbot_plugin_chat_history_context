from __future__ import annotations

import inspect
import sys
from typing import Any


INTERFACE_NAME = "historywatch"
HISTORYWATCH_API_VERSION = 1
_provider: Any | None = None


def _matching_modules() -> list[Any]:
    current = sys.modules.get(__name__)
    return [
        module
        for name, module in list(sys.modules.items())
        if module is not None
        and (
            name == "astrbot_plugin_chat_history_context.integration"
            or name.endswith(".astrbot_plugin_chat_history_context.integration")
        )
        and module is not current
    ]


def register_provider(provider: Any) -> None:
    global _provider
    _provider = provider
    for module in _matching_modules():
        setattr(module, "_provider", provider)


def unregister_provider(provider: Any) -> None:
    global _provider
    if _provider is provider:
        _provider = None
    for module in _matching_modules():
        if getattr(module, "_provider", None) is provider:
            setattr(module, "_provider", None)


async def query_history(
    *,
    umo: str,
    logical_group_id: str = "",
    start_ts: float,
    end_ts: float,
    limit: int = 100,
    exclude_row_id: int | None = None,
) -> list[dict[str, Any]]:
    provider = _provider
    if provider is None:
        return []
    method = getattr(provider, "query_history", None)
    if not callable(method):
        return []
    result = method(
        umo=umo,
        logical_group_id=logical_group_id,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
        exclude_row_id=exclude_row_id,
    )
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (list, tuple)):
        return []
    return [dict(item) for item in result if isinstance(item, dict)]


async def sender_aliases(logical_group_id: str) -> list[dict[str, Any]]:
    provider = _provider
    if provider is None:
        return []
    method = getattr(provider, "sender_aliases", None)
    if not callable(method):
        return []
    result = method(logical_group_id)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (list, tuple)):
        return []
    return [dict(item) for item in result if isinstance(item, dict)]


class HistoryWatchAPI:
    """Stable, versioned cross-plugin interface exposed by HistoryWatch."""

    name = INTERFACE_NAME
    version = HISTORYWATCH_API_VERSION

    @property
    def available(self) -> bool:
        provider = _provider
        return provider is not None and callable(
            getattr(provider, "query_history", None)
        )

    @property
    def capabilities(self) -> dict[str, bool]:
        """Report optional methods without making callers introspect the provider."""
        provider = _provider
        return {
            "query_history": bool(
                provider is not None
                and callable(getattr(provider, "query_history", None))
            ),
            "sender_aliases": bool(
                provider is not None
                and callable(getattr(provider, "sender_aliases", None))
            ),
        }

    async def query_history(
        self,
        *,
        umo: str,
        logical_group_id: str = "",
        start_ts: float,
        end_ts: float,
        limit: int = 100,
        exclude_row_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await query_history(
            umo=umo,
            logical_group_id=logical_group_id,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            exclude_row_id=exclude_row_id,
        )

    async def sender_aliases(
        self, logical_group_id: str
    ) -> list[dict[str, Any]]:
        return await sender_aliases(logical_group_id)


_api = HistoryWatchAPI()


def get_historywatch_api(*, minimum_version: int = 1) -> HistoryWatchAPI:
    """Return the supported API or fail explicitly on a version mismatch."""

    try:
        requested = int(minimum_version)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_version must be an integer") from exc
    if requested < 1:
        raise ValueError("minimum_version must be at least 1")
    if requested > HISTORYWATCH_API_VERSION:
        raise RuntimeError(
            "HistoryWatch interface version mismatch: "
            f"requested >= {requested}, available {HISTORYWATCH_API_VERSION}"
        )
    return _api


def api_status() -> dict[str, Any]:
    return {
        "name": INTERFACE_NAME,
        "version": HISTORYWATCH_API_VERSION,
        "available": _api.available,
        "capabilities": _api.capabilities,
    }


# Register the canonical leaf module too.  Without this, AstrBot's dynamic
# plugin namespace can cause a second copy of this module (and therefore an
# empty provider registry) to be imported by another plugin.
sys.modules.setdefault(
    "astrbot_plugin_chat_history_context.integration", sys.modules[__name__]
)
