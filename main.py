from __future__ import annotations

import asyncio
import inspect
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
        self._last_cleanup_at = 0.0
        self.timezone = self._load_timezone()
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
        return group_is_monitored(
            group_id=self._safe_text(event.get_group_id()),
            umo=self._safe_text(event.unified_msg_origin),
            platform_names=platform_names,
            targets=self._group_targets(),
            listen_all=bool(self.config.get("listen_all_groups", False)),
        )

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
        target = self._safe_text(event.unified_msg_origin)
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
        removable = {group_id, umo}
        for platform in platform_names:
            removable.update({f"{platform}:{group_id}", f"{platform}/{group_id}"})
        targets = [
            target for target in self._group_targets()
            if target.casefold() not in removable
        ]
        await self._save_group_targets(targets)
        still_all = bool(self.config.get("listen_all_groups", False))
        yield event.plain_result(
            "已从特定群监听白名单移除当前群。"
            + ("但“监听全部群”仍处于开启状态。" if still_all else "")
        )

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=-200)
    async def capture_group_message(self, event: AstrMessageEvent) -> None:
        if not bool(self.config.get("enabled", True)):
            return
        if not self._is_monitored_group(event):
            return
        if (
            not bool(self.config.get("include_self_messages", True))
            and self._safe_text(event.get_sender_id())
            == self._safe_text(event.get_self_id())
        ):
            return

        content = self._format_message(event)
        if not content:
            return

        now_ts = time.time()
        message_id = self._safe_text(
            getattr(event.message_obj, "message_id", "")
        ) or None
        async with self._write_lock:
            row_id = await asyncio.to_thread(
                self.store.append,
                umo=self._safe_text(event.unified_msg_origin),
                ts=now_ts,
                sender_id=self._safe_text(event.get_sender_id()),
                sender_name=self._safe_text(event.get_sender_name()) or "未知成员",
                content=content,
                platform=self._safe_text(event.get_platform_name()),
                group_id=self._safe_text(event.get_group_id()),
                message_id=message_id,
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
        block = format_history_block(
            records,
            scope,
            self.timezone,
            max_messages=max_messages,
            max_chars=max_chars,
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
