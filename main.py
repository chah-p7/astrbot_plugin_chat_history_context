from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import (
    At,
    AtAll,
    Face,
    File,
    Forward,
    Image,
    Plain,
    Record,
    Reply,
    Video,
)
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart

from .core import (
    TimeScope,
    build_identity_system_prompt,
    format_history_block,
    group_is_monitored,
    is_explicit_history_request,
    parse_time_scope,
    parse_group_targets,
)
from .storage import ChatHistoryStore
from .integration import register_provider, unregister_provider


PLUGIN_NAME = "astrbot_plugin_chat_history_context"
_EVENT_ROW_ID_KEY = "_chat_history_context_row_id"


class ChatHistoryContextPlugin(Star):
    """按需记录并注入未唤醒的群聊消息。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.store = ChatHistoryStore(self.data_dir / "history.sqlite3")
        self._write_lock = asyncio.Lock()
        self._backfilled_logical_groups: dict[str, set[str]] = {}
        self._last_cleanup_at = 0.0
        self.timezone = self._load_timezone()
        register_provider(self)
        logger.info(
            "[聊天记录上下文] 插件已加载：默认回溯 %s 小时，保留 %s 天，数据库 %s",
            self._default_hours(),
            self._retention_days(),
            self.store.path,
        )

    def _load_timezone(self) -> ZoneInfo:
        name = str(self.config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
        try:
            return ZoneInfo(name)
        except Exception:
            logger.warning("[聊天记录上下文] 无效时区 %s，已使用 Asia/Shanghai", name)
            return ZoneInfo("Asia/Shanghai")

    def _default_hours(self) -> float:
        try:
            return max(1 / 60, float(self.config.get("default_hours", 6)))
        except (TypeError, ValueError):
            return 6

    def _retention_days(self) -> float:
        try:
            return max(1, float(self.config.get("retention_days", 14)))
        except (TypeError, ValueError):
            return 14

    def _max_message_chars(self) -> int:
        try:
            return max(200, int(self.config.get("max_message_chars", 4000)))
        except (TypeError, ValueError):
            return 4000

    def _max_injected_messages(self) -> int:
        try:
            return max(1, int(self.config.get("max_injected_messages", 600)))
        except (TypeError, ValueError):
            return 600

    def _max_injected_chars(self) -> int:
        try:
            return max(1000, int(self.config.get("max_injected_chars", 100000)))
        except (TypeError, ValueError):
            return 100000

    def _continuous_context_hours(self) -> float:
        try:
            return max(
                1 / 60,
                float(self.config.get("continuous_context_hours", 2)),
            )
        except (TypeError, ValueError):
            return 2

    def _continuous_max_messages(self) -> int:
        try:
            return max(
                1,
                int(self.config.get("continuous_max_messages", 100)),
            )
        except (TypeError, ValueError):
            return 100

    def _continuous_max_chars(self) -> int:
        try:
            return max(
                1000,
                int(self.config.get("continuous_max_chars", 20000)),
            )
        except (TypeError, ValueError):
            return 20000

    def _extra_trigger_phrases(self) -> list[str]:
        raw = str(self.config.get("extra_trigger_phrases", "") or "")
        return [part.strip() for part in re.split(r"[,，\n]", raw) if part.strip()]

    def _group_targets(self) -> tuple[str, ...]:
        return parse_group_targets(self.config.get("listen_groups", ""))

    def _botmesh_config(self) -> dict[str, object]:
        """Read BotMesh bindings when its dynamic integration is unavailable."""
        data_dir = self.data_dir
        candidates = (
            data_dir / "astrbot_plugin_botmesh_config.json",
            data_dir.parent / "config" / "astrbot_plugin_botmesh_config.json",
            data_dir.parent.parent / "config" / "astrbot_plugin_botmesh_config.json",
        )
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
        return {}

    def _botmesh_scope_fallback(self, umo: str) -> dict[str, object]:
        config = self._botmesh_config()
        parts = self._safe_text(umo).split(":", 2)
        if len(parts) != 3:
            return {}
        platform_id, _, raw_group_id = parts
        bots = {
            self._safe_text(item.get("bot_id")): item
            for item in config.get("bots", [])
            if isinstance(item, dict) and self._safe_text(item.get("bot_id"))
        }
        for binding in config.get("group_bindings", []):
            if not isinstance(binding, dict):
                continue
            bot = bots.get(self._safe_text(binding.get("bot_id")), {})
            if (
                self._safe_text(bot.get("platform_id")) != platform_id
                or self._safe_text(binding.get("platform_group_id")) != raw_group_id
            ):
                continue
            logical_group_id = self._safe_text(binding.get("group_id"))
            if not logical_group_id:
                return {}
            selectors = [f"botmesh:{logical_group_id}"]
            for candidate in config.get("group_bindings", []):
                if not isinstance(candidate, dict):
                    continue
                if self._safe_text(candidate.get("group_id")) != logical_group_id:
                    continue
                candidate_bot = bots.get(self._safe_text(candidate.get("bot_id")), {})
                candidate_raw = self._safe_text(candidate.get("platform_group_id"))
                candidate_platform = self._safe_text(candidate_bot.get("platform_id"))
                if not candidate_raw or not candidate_platform:
                    continue
                selectors.extend(
                    (
                        f"{candidate_platform}:{candidate_raw}",
                        f"{candidate_platform}/{candidate_raw}",
                        f"{candidate_platform}:GroupMessage:{candidate_raw}",
                    )
                )
            return {
                "selector": f"botmesh:{logical_group_id}",
                "logical_group_id": logical_group_id,
                "selectors": list(dict.fromkeys(selectors)),
            }
        return {}

    def _botmesh_management_labels_fallback(self) -> dict[str, dict[str, str]]:
        config = self._botmesh_config()
        bots = {
            self._safe_text(item.get("bot_id")): item
            for item in config.get("bots", [])
            if isinstance(item, dict) and self._safe_text(item.get("bot_id"))
        }
        labels: dict[str, dict[str, str]] = {"scope_groups": {}}
        for binding in config.get("group_bindings", []):
            if not isinstance(binding, dict):
                continue
            group_id = self._safe_text(binding.get("group_id"))
            bot = bots.get(self._safe_text(binding.get("bot_id")), {})
            platform_id = self._safe_text(bot.get("platform_id"))
            raw_group_id = self._safe_text(binding.get("platform_group_id"))
            if not group_id or not platform_id or not raw_group_id:
                continue
            for selector in (
                f"{platform_id}:{raw_group_id}",
                f"{platform_id}/{raw_group_id}",
                f"{platform_id}:GroupMessage:{raw_group_id}",
            ):
                labels["scope_groups"][selector] = group_id
        return labels

    def _botmesh_history_scope(self, event: AstrMessageEvent) -> dict[str, object]:
        """Read optional logical-group aliases without taking a hard dependency."""
        try:
            integration = importlib.import_module("astrbot_plugin_botmesh.integration")
            method = getattr(integration, "get_chat_history_scope", None)
            if not callable(method):
                return self._botmesh_scope_fallback(
                    self._safe_text(event.unified_msg_origin)
                )
            result = method(
                umo=self._safe_text(event.unified_msg_origin),
                event=event,
            )
            mapped = dict(result) if isinstance(result, dict) else {}
            return mapped or self._botmesh_scope_fallback(
                self._safe_text(event.unified_msg_origin)
            )
        except (ImportError, AttributeError):
            return self._botmesh_scope_fallback(
                self._safe_text(event.unified_msg_origin)
            )
        except Exception as exc:
            logger.debug("[聊天记录上下文] 读取 BotMesh 群映射失败: %s", exc)
        return self._botmesh_scope_fallback(self._safe_text(event.unified_msg_origin))

    def _botmesh_history_scope_for_umo(self, umo: str) -> dict[str, object]:
        try:
            integration = importlib.import_module("astrbot_plugin_botmesh.integration")
            method = getattr(integration, "get_chat_history_scope", None)
            if not callable(method):
                return self._botmesh_scope_fallback(self._safe_text(umo))
            result = method(umo=self._safe_text(umo), event=None)
            mapped = dict(result) if isinstance(result, dict) else {}
            return mapped or self._botmesh_scope_fallback(self._safe_text(umo))
        except Exception as exc:
            logger.debug("[聊天记录上下文] 按 UMO 读取 BotMesh 群映射失败: %s", exc)
        return self._botmesh_scope_fallback(self._safe_text(umo))

    def _botmesh_history_scope_for_logical_group(
        self,
        logical_group_id: str,
    ) -> dict[str, object]:
        """Resolve every persisted UMO when a caller has no current UMO.

        The memory manager summarizes a logical group from a web request, so it
        cannot provide one platform UMO.  The normal event-scoped resolver is
        intentionally unable to resolve an empty UMO; use BotMesh's label map
        to obtain the bound full UMO selectors for this case.
        """
        target = self._safe_text(logical_group_id)
        if not target:
            return {}
        try:
            integration = importlib.import_module("astrbot_plugin_botmesh.integration")
            method = getattr(integration, "get_management_labels", None)
            if not callable(method):
                labels = self._botmesh_management_labels_fallback()
                scope_groups = labels.get("scope_groups", {})
            else:
                labels = method()
                if not isinstance(labels, dict):
                    labels = self._botmesh_management_labels_fallback()
            scope_groups = labels.get("scope_groups", {})
            if not isinstance(scope_groups, dict):
                scope_groups = self._botmesh_management_labels_fallback().get(
                    "scope_groups", {}
                )
            if not any(
                self._safe_text(group_id) == target
                and ":GroupMessage:" in self._safe_text(scope_id)
                for scope_id, group_id in scope_groups.items()
            ):
                scope_groups = self._botmesh_management_labels_fallback().get(
                    "scope_groups", {}
                )
            selectors = [
                self._safe_text(scope_id)
                for scope_id, group_id in scope_groups.items()
                if self._safe_text(group_id) == target
                and ":GroupMessage:" in self._safe_text(scope_id)
            ]
            return {
                "logical_group_id": target,
                "selectors": list(dict.fromkeys(item for item in selectors if item)),
            }
        except Exception as exc:
            logger.debug(
                "[聊天记录上下文] 按逻辑群读取 BotMesh 群映射失败: %s",
                exc,
            )
            labels = self._botmesh_management_labels_fallback()
            scope_groups = labels.get("scope_groups", {})
            selectors = [
                self._safe_text(scope_id)
                for scope_id, group_id in scope_groups.items()
                if self._safe_text(group_id) == target
                and ":GroupMessage:" in self._safe_text(scope_id)
            ]
            return {
                "logical_group_id": target,
                "selectors": list(dict.fromkeys(item for item in selectors if item)),
            }

    @staticmethod
    def _scope_selectors(scope: dict[str, object]) -> tuple[str, ...]:
        raw = scope.get("selectors", ())
        if not isinstance(raw, (list, tuple, set)):
            return ()
        return tuple(str(item or "").strip() for item in raw if str(item or "").strip())

    @staticmethod
    def _history_self_aliases(scope: dict[str, object]) -> tuple[str, ...]:
        """收集当前 Bot 在群里的可识别别名，用于判断消息是否直接 @ 到自己。"""
        aliases: set[str] = set()
        bot_id = str(scope.get("bot_id") or "").strip()
        if bot_id:
            aliases.add(bot_id)
            aliases.add(bot_id.removeprefix("bot_"))
        account_id = str(scope.get("account_id") or "").strip()
        if account_id:
            aliases.add(account_id)
        label = str(scope.get("bot_display_name") or "").strip()
        if label:
            aliases.add(label)
        identity = scope.get("identity_state")
        if isinstance(identity, dict):
            for key in (
                "self_identity",
                "body_identity",
                "soul_identity",
                "memory_key",
                "name",
                "display_name",
                "nickname",
                "account_label",
            ):
                value = str(identity.get(key) or "").strip()
                if value:
                    aliases.add(value)
        aliases.discard("")
        return tuple(sorted(aliases))

    async def _ensure_logical_backfill(
        self,
        *,
        logical_group_id: str,
        current_umo: str,
        scope: dict[str, object],
    ) -> None:
        if not logical_group_id:
            return
        umos = [self._safe_text(current_umo)]
        umos.extend(
            selector
            for selector in self._scope_selectors(scope)
            if ":GroupMessage:" in selector
        )
        umos = list(dict.fromkeys(item for item in umos if item))
        # Do not permanently mark a group as backfilled when no resolver was
        # available; a later API call may be able to provide its selectors.
        if not umos:
            return
        known_umos = self._backfilled_logical_groups.setdefault(logical_group_id, set())
        if set(umos).issubset(known_umos):
            return
        changed = await asyncio.to_thread(
            self.store.backfill_logical_group,
            logical_group_id=logical_group_id,
            umos=umos,
        )
        known_umos.update(umos)
        if changed:
            logger.info(
                "[聊天记录上下文] 已将 %d 条旧记录迁入逻辑群 %s 并建立去重事件",
                changed,
                logical_group_id,
            )

    def _is_monitored_group(self, event: AstrMessageEvent) -> bool:
        platform_names: list[str] = []
        for getter_name in ("get_platform_id", "get_platform_name"):
            getter = getattr(event, getter_name, None)
            if not callable(getter):
                continue
            try:
                value = self._safe_text(getter())
            except Exception:
                value = ""
            if value:
                platform_names.append(value)
        scope = self._botmesh_history_scope(event)
        return group_is_monitored(
            group_id=self._safe_text(event.get_group_id()),
            umo=self._safe_text(event.unified_msg_origin),
            platform_names=platform_names,
            targets=self._group_targets(),
            extra_candidates=self._scope_selectors(scope),
            listen_all=bool(self.config.get("listen_all_groups", False)),
        )

    async def _normalize_botmesh_content(
        self,
        event: AstrMessageEvent,
        content: str,
    ) -> str:
        """Strip a verified BotMesh transport frame before persisting its body."""
        record = await self._normalize_botmesh_record(event, content)
        return str(record.get("content", "") or content)

    async def _normalize_botmesh_record(
        self,
        event: AstrMessageEvent,
        content: str,
    ) -> dict[str, str]:
        """Keep BotMesh's verified sender identity together with the visible body."""
        fallback = {"content": content}
        try:
            integration = importlib.import_module("astrbot_plugin_botmesh.integration")
            record_method = getattr(
                integration,
                "normalize_chat_history_record",
                None,
            )
            if callable(record_method):
                result = record_method(
                    umo=self._safe_text(event.unified_msg_origin),
                    content=content,
                    event=event,
                )
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    normalized = {
                        key: self._safe_text(value)
                        for key, value in result.items()
                        if key
                        in {"content", "sender_id", "sender_name", "source_bot_id"}
                    }
                    normalized.setdefault("content", content)
                    return normalized
            method = getattr(integration, "normalize_chat_history_message", None)
            if not callable(method):
                return fallback
            result = method(
                umo=self._safe_text(event.unified_msg_origin),
                content=content,
                event=event,
            )
            if inspect.isawaitable(result):
                result = await result
            normalized = str(result or "").strip()
            fallback["content"] = normalized or content
            return fallback
        except (ImportError, AttributeError):
            return fallback
        except Exception as exc:
            logger.debug("[聊天记录上下文] 清理 BotMesh 展示帧失败: %s", exc)
            return fallback

    async def _save_group_targets(self, targets: list[str]) -> None:
        self.config["listen_groups"] = "\n".join(targets)
        result = self.config.save_config()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _safe_text(value) -> str:
        return str(value or "").strip()

    def _format_message(self, event: AstrMessageEvent) -> str:
        parts: list[str] = []
        for component in event.get_messages():
            if isinstance(component, Plain):
                parts.append(component.text)
            elif isinstance(component, Image):
                parts.append("[图片]")
            elif isinstance(component, Record):
                parts.append("[语音]")
            elif isinstance(component, Video):
                parts.append("[视频]")
            elif isinstance(component, File):
                name = self._safe_text(getattr(component, "name", ""))
                parts.append(f"[文件: {name}]" if name else "[文件]")
            elif isinstance(component, Forward):
                parts.append("[转发消息]")
            elif isinstance(component, AtAll):
                parts.append("[@全体成员]")
            elif isinstance(component, At):
                name = self._safe_text(getattr(component, "name", ""))
                qq = self._safe_text(getattr(component, "qq", ""))
                parts.append(f"[@{name or qq or '成员'}]")
            elif isinstance(component, Face):
                face_id = self._safe_text(getattr(component, "id", ""))
                parts.append(f"[表情: {face_id}]" if face_id else "[表情]")
            elif isinstance(component, Reply):
                reply_text = self._safe_text(getattr(component, "message_str", ""))
                reply_sender = self._safe_text(
                    getattr(component, "sender_nickname", "")
                )
                if reply_text:
                    reply_text = re.sub(r"\s+", " ", reply_text)[:300]
                    prefix = f"{reply_sender}: " if reply_sender else ""
                    parts.append(f"[回复 {prefix}{reply_text}]")
                else:
                    parts.append("[回复消息]")
            else:
                parts.append(f"[{component.__class__.__name__}]")

        content = "".join(parts).strip()
        content = re.sub(r"\r\n?", "\n", content)
        limit = self._max_message_chars()
        if len(content) > limit:
            content = content[:limit] + "…[消息截断]"
        return content

    async def _maybe_prune(self, now_ts: float) -> None:
        if now_ts - self._last_cleanup_at < 600:
            return
        self._last_cleanup_at = now_ts
        cutoff = now_ts - self._retention_days() * 86400
        deleted = await asyncio.to_thread(self.store.prune_before, cutoff)
        if deleted:
            logger.info("[聊天记录上下文] 已清理 %d 条过期群消息", deleted)

    @filter.command_group("historywatch")
    def historywatch(self):
        """管理持续监听的群聊白名单。"""
        pass

    @filter.permission_type(filter.PermissionType.ADMIN)
    @historywatch.command("status")
    async def historywatch_status(self, event: AstrMessageEvent):
        if not event.get_group_id():
            yield event.plain_result("该命令只能在群聊中使用。")
            return
        monitored = self._is_monitored_group(event)
        count = await asyncio.to_thread(
            self.store.count_since,
            umo=self._safe_text(event.unified_msg_origin),
            start_ts=time.time() - 86400,
        )
        yield event.plain_result(
            "聊天记录持续监听状态：\n"
            f"- 当前群：{'已监听' if monitored else '未监听'}\n"
            f"- 群 ID：{self._safe_text(event.get_group_id())}\n"
            f"- UMO：{self._safe_text(event.unified_msg_origin)}\n"
            f"- 最近 24 小时已保存：{count} 条\n"
            f"- 全群监听：{'开启' if self.config.get('listen_all_groups', False) else '关闭'}\n"
            f"- 每轮自动上下文：{'开启' if self.config.get('continuous_context_enabled', True) else '关闭'}\n"
            f"- 自动上下文窗口：最近 {self._continuous_context_hours():g} 小时"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @historywatch.command("add")
    async def historywatch_add(self, event: AstrMessageEvent):
        if not event.get_group_id():
            yield event.plain_result("该命令只能在要监听的群聊中使用。")
            return
        scope = self._botmesh_history_scope(event)
        target = self._safe_text(scope.get("selector")) or self._safe_text(
            event.unified_msg_origin
        )
        targets = list(self._group_targets())
        if target not in targets:
            targets.append(target)
            await self._save_group_targets(targets)
        yield event.plain_result(
            "已持续监听当前群的全部后续消息，并在每轮 LLM 请求中自动加入最近群聊上下文。\n"
            f"保存标识：{target}\n"
            "可用 /historywatch status 查看记录状态。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @historywatch.command("remove")
    async def historywatch_remove(self, event: AstrMessageEvent):
        if not event.get_group_id():
            yield event.plain_result("该命令只能在群聊中使用。")
            return
        group_id = self._safe_text(event.get_group_id()).casefold()
        umo = self._safe_text(event.unified_msg_origin).casefold()
        platform_names: list[str] = []
        for getter_name in ("get_platform_id", "get_platform_name"):
            getter = getattr(event, getter_name, None)
            if callable(getter):
                try:
                    value = self._safe_text(getter()).casefold()
                except Exception:
                    value = ""
                if value:
                    platform_names.append(value)
        scope = self._botmesh_history_scope(event)
        removable = {group_id, umo}
        for platform in platform_names:
            removable.update({f"{platform}:{group_id}", f"{platform}/{group_id}"})
        removable.update(self._scope_selectors(scope))
        selector = self._safe_text(scope.get("selector")).casefold()
        if selector:
            removable.add(selector)
        targets = [
            target
            for target in self._group_targets()
            if not group_is_monitored(
                group_id=group_id,
                umo=umo,
                platform_names=platform_names,
                targets=(target,),
                extra_candidates=removable,
                listen_all=False,
            )
        ]
        await self._save_group_targets(targets)
        still_all = bool(self.config.get("listen_all_groups", False))
        yield event.plain_result(
            "已从特定群监听白名单移除当前群。"
            + ("但“监听全部群”仍处于开启状态。" if still_all else "")
        )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=200)
    async def capture_group_message(self, event: AstrMessageEvent) -> None:
        if not bool(self.config.get("enabled", True)):
            return
        if not self._is_monitored_group(event):
            return
        content = self._format_message(event)
        if not content:
            return
        normalized = await self._normalize_botmesh_record(event, content)
        scope = self._botmesh_history_scope(event)
        if not bool(self.config.get("include_self_messages", True)):
            sender_is_self = self._safe_text(event.get_sender_id()) == self._safe_text(
                event.get_self_id()
            )
            normalized_bot_id = self._safe_text(normalized.get("source_bot_id"))
            scope_bot_id = self._safe_text(scope.get("bot_id"))
            if sender_is_self or (
                normalized_bot_id
                and scope_bot_id
                and normalized_bot_id == scope_bot_id
            ):
                return
        logical_group_id = self._safe_text(scope.get("logical_group_id"))
        await self._ensure_logical_backfill(
            logical_group_id=logical_group_id,
            current_umo=self._safe_text(event.unified_msg_origin),
            scope=scope,
        )
        content = self._safe_text(normalized.get("content")) or content
        sender_id = self._safe_text(normalized.get("sender_id")) or self._safe_text(
            event.get_sender_id()
        )
        sender_name = self._safe_text(
            normalized.get("sender_name")
        ) or self._safe_text(event.get_sender_name()) or "未知成员"

        now_ts = time.time()
        message_id = self._safe_text(
            getattr(event.message_obj, "message_id", "")
        ) or None
        async with self._write_lock:
            row_id = await asyncio.to_thread(
                self.store.append,
                umo=self._safe_text(event.unified_msg_origin),
                ts=now_ts,
                sender_id=sender_id,
                sender_name=sender_name,
                content=content,
                platform=self._safe_text(event.get_platform_name()),
                group_id=self._safe_text(event.get_group_id()),
                message_id=message_id,
                logical_group_id=logical_group_id,
                source_bot_id=self._safe_text(normalized.get("source_bot_id")),
            )
            await self._maybe_prune(now_ts)
        event.set_extra(_EVENT_ROW_ID_KEY, row_id)

    @filter.on_llm_request(priority=80)
    async def inject_history_on_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if not bool(self.config.get("enabled", True)):
            return
        if not event.get_group_id():
            return
        if not self._is_monitored_group(event):
            return

        current_text = self._safe_text(event.get_message_str()) or self._safe_text(
            req.prompt
        )
        explicit_request = is_explicit_history_request(
            current_text,
            self._extra_trigger_phrases(),
        )

        now = datetime.now(self.timezone)
        if explicit_request:
            scope = parse_time_scope(
                current_text,
                now,
                default_hours=self._default_hours(),
                retention_days=self._retention_days(),
            )
            max_messages = self._max_injected_messages()
            max_chars = self._max_injected_chars()
        else:
            if not bool(self.config.get("continuous_context_enabled", True)):
                return
            window_seconds = self._continuous_context_hours() * 3600
            retention_seconds = self._retention_days() * 86400
            effective_seconds = min(window_seconds, retention_seconds)
            scope = TimeScope(
                start_ts=now.timestamp() - effective_seconds,
                end_ts=now.timestamp(),
                label=f"最近{effective_seconds / 3600:g}小时（每轮自动上下文）",
                reason="当前群位于持续上下文白名单",
                clipped_by_retention=window_seconds > retention_seconds,
            )
            max_messages = self._continuous_max_messages()
            max_chars = self._continuous_max_chars()
        exclude_row_id = event.get_extra(_EVENT_ROW_ID_KEY, None)
        if not isinstance(exclude_row_id, int):
            exclude_row_id = None

        records = await asyncio.to_thread(
            self.store.query,
            umo=self._safe_text(event.unified_msg_origin),
            start_ts=scope.start_ts,
            end_ts=scope.end_ts,
            exclude_row_id=exclude_row_id,
        )
        if not records and not explicit_request:
            return
        identity_prompt = build_identity_system_prompt(
            self._safe_text(event.get_sender_id())
        )
        existing_system_prompt = self._safe_text(req.system_prompt)
        if "[群聊用户身份识别规则]" not in existing_system_prompt:
            req.system_prompt = (
                f"{existing_system_prompt}\n\n{identity_prompt}".strip()
            )
        botmesh_scope = self._botmesh_history_scope(event)
        self_aliases = self._history_self_aliases(botmesh_scope)
        block = format_history_block(
            records,
            scope,
            self.timezone,
            max_messages=max_messages,
            max_chars=max_chars,
            self_aliases=self_aliases,
        )
        if req.extra_user_content_parts is None:
            req.extra_user_content_parts = []
        req.extra_user_content_parts.append(TextPart(text=block).mark_as_temp())
        logger.info(
            "[聊天记录上下文] 已为群 %s 注入 %d 条消息，范围：%s",
            self._safe_text(event.get_group_id()),
            len(records),
            scope.label,
        )

    async def query_history(
        self,
        *,
        umo: str,
        logical_group_id: str,
        start_ts: float,
        end_ts: float,
        limit: int,
        exclude_row_id: int | None = None,
    ) -> list[dict[str, object]]:
        if logical_group_id:
            scope = self._botmesh_history_scope_for_umo(umo)
            if (
                self._safe_text(scope.get("logical_group_id"))
                != self._safe_text(logical_group_id)
                or not self._scope_selectors(scope)
            ):
                scope = self._botmesh_history_scope_for_logical_group(
                    logical_group_id
                )
            await self._ensure_logical_backfill(
                logical_group_id=logical_group_id,
                current_umo=umo,
                scope=scope,
            )
            records = await asyncio.to_thread(
                self.store.query_logical,
                logical_group_id=logical_group_id,
                start_ts=start_ts,
                end_ts=end_ts,
                exclude_row_id=exclude_row_id,
                limit=limit,
            )
        else:
            records = await asyncio.to_thread(
                self.store.query,
                umo=umo,
                start_ts=start_ts,
                end_ts=end_ts,
                exclude_row_id=exclude_row_id,
            )
            records = records[-max(1, int(limit)) :]
        return [
            {
                "row_id": record.row_id,
                "umo": record.umo,
                "ts": record.ts,
                "sender_id": record.sender_id,
                "canonical_sender_id": (
                    record.canonical_sender_id or record.sender_id
                ),
                "sender_name": record.sender_name,
                "content": record.content,
                "logical_group_id": record.logical_group_id,
                "logical_event_id": record.logical_event_id,
                "source_bot_id": record.source_bot_id,
            }
            for record in records
        ]

    async def sender_aliases(self, logical_group_id: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(
            self.store.aliases_for_group,
            logical_group_id,
        )

    async def terminate(self) -> None:
        unregister_provider(self)
        logger.info("[聊天记录上下文] 插件已停止")
