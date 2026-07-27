from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _decorator(*_args, **_kwargs):
    def decorate(function):
        function._filter_kwargs = dict(_kwargs)
        return function

    return decorate


def _command_group(_name):
    def decorate(function):
        function.command = lambda *_args, **_kwargs: _decorator()
        return function

    return decorate


class _Star:
    def __init__(self, context):
        self.context = context


class _StarTools:
    root = Path(tempfile.gettempdir())

    @classmethod
    def get_data_dir(cls, _name):
        path = cls.root / "plugin_data"
        path.mkdir(parents=True, exist_ok=True)
        return path


class Plain:
    def __init__(self, text=""):
        self.text = text


def _component(name):
    return type(name, (), {})


class _TextPart:
    def __init__(self, text):
        self.text = text
        self.temporary = False

    def mark_as_temp(self):
        self.temporary = True
        return self


def _install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    astrbot.__path__ = []
    api = types.ModuleType("astrbot.api")
    api.__path__ = []
    api.AstrBotConfig = dict
    api.logger = _Logger()

    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = types.SimpleNamespace(
        command_group=_command_group,
        permission_type=_decorator,
        event_message_type=_decorator,
        on_llm_request=_decorator,
        PermissionType=types.SimpleNamespace(ADMIN="admin"),
        EventMessageType=types.SimpleNamespace(GROUP_MESSAGE="group"),
    )

    components = types.ModuleType("astrbot.api.message_components")
    components.Plain = Plain
    for name in ("At", "AtAll", "Face", "File", "Forward", "Image", "Record", "Reply", "Video"):
        setattr(components, name, _component(name))

    provider = types.ModuleType("astrbot.api.provider")
    provider.ProviderRequest = object
    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = _Star
    star.StarTools = _StarTools

    core = types.ModuleType("astrbot.core")
    core.__path__ = []
    agent = types.ModuleType("astrbot.core.agent")
    agent.__path__ = []
    message = types.ModuleType("astrbot.core.agent.message")
    message.TextPart = _TextPart

    sys.modules.update(
        {
            "astrbot": astrbot,
            "astrbot.api": api,
            "astrbot.api.event": event,
            "astrbot.api.message_components": components,
            "astrbot.api.provider": provider,
            "astrbot.api.star": star,
            "astrbot.core": core,
            "astrbot.core.agent": agent,
            "astrbot.core.agent.message": message,
        }
    )


_install_astrbot_stubs()

from astrbot_plugin_chat_history_context.main import ChatHistoryContextPlugin


class _Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = 0

    def save_config(self):
        self.saved += 1


class _MessageObject:
    def __init__(self, message_id):
        self.message_id = message_id


class _Event:
    def __init__(
        self,
        *,
        group_id="100",
        umo="qq:group:100",
        text="大家好",
        message_id="m1",
        platform_id="onebot_main",
        platform_name="aiocqhttp",
    ):
        self.unified_msg_origin = umo
        self.message_obj = _MessageObject(message_id)
        self._group_id = group_id
        self._text = text
        self._platform_id = platform_id
        self._platform_name = platform_name
        self._extra = {}

    def get_group_id(self):
        return self._group_id

    def get_platform_id(self):
        return self._platform_id

    def get_platform_name(self):
        return self._platform_name

    def get_sender_id(self):
        return "20001"

    def get_self_id(self):
        return "10001"

    def get_sender_name(self):
        return "群友"

    def get_messages(self):
        return [Plain(self._text)]

    def get_message_str(self):
        return self._text

    def set_extra(self, key, value):
        self._extra[key] = value

    def get_extra(self, key, default=None):
        return self._extra.get(key, default)

    def plain_result(self, text):
        return text


class PluginRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        _StarTools.root = Path(self.temp_dir.name)
        self.config = _Config(
            enabled=True,
            listen_all_groups=False,
            listen_groups="",
            include_self_messages=True,
            timezone="Asia/Shanghai",
        )
        self.plugin = ChatHistoryContextPlugin(object(), self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_add_current_group_then_capture_only_that_group(self):
        event = _Event()
        replies = [item async for item in self.plugin.historywatch_add(event)]

        self.assertEqual(self.config.saved, 1)
        self.assertIn("qq:group:100", self.config["listen_groups"])
        self.assertIn("已持续监听", replies[0])

        await self.plugin.capture_group_message(event)
        await self.plugin.capture_group_message(
            _Event(group_id="999", umo="qq:group:999", message_id="m2")
        )
        self.assertEqual(
            self.plugin.store.count_since(umo="qq:group:100", start_ts=0),
            1,
        )
        self.assertEqual(
            self.plugin.store.count_since(umo="qq:group:999", start_ts=0),
            0,
        )

    async def test_history_is_injected_as_temporary_content(self):
        self.config["listen_groups"] = "qq:group:100"
        await self.plugin.capture_group_message(_Event())
        request_event = _Event(text="看看最近一小时聊天记录", message_id="m2")
        req = types.SimpleNamespace(
            prompt=request_event.get_message_str(),
            system_prompt="原系统提示",
            extra_user_content_parts=[],
        )

        await self.plugin.inject_history_on_request(request_event, req)

        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertTrue(req.extra_user_content_parts[0].temporary)
        self.assertIn("大家好", req.extra_user_content_parts[0].text)
        self.assertIn("群聊用户身份识别规则", req.system_prompt)

    async def test_monitored_group_gets_history_on_ordinary_llm_request(self):
        self.config.update(
            listen_groups="qq:group:100",
            continuous_context_enabled=True,
            continuous_context_hours=2,
            continuous_max_messages=100,
            continuous_max_chars=20000,
        )
        await self.plugin.capture_group_message(_Event(text="不需要艾特也会被记录"))
        request_event = _Event(text="你觉得呢？", message_id="m2")
        req = types.SimpleNamespace(
            prompt=request_event.get_message_str(),
            system_prompt="",
            extra_user_content_parts=[],
        )

        await self.plugin.inject_history_on_request(request_event, req)

        self.assertEqual(len(req.extra_user_content_parts), 1)
        self.assertTrue(req.extra_user_content_parts[0].temporary)
        self.assertIn("不需要艾特也会被记录", req.extra_user_content_parts[0].text)
        self.assertIn("每轮自动上下文", req.extra_user_content_parts[0].text)

    async def test_unmonitored_group_never_gets_continuous_context(self):
        self.config["listen_groups"] = "qq:group:100"
        request_event = _Event(
            group_id="999",
            umo="qq:group:999",
            text="你觉得呢？",
        )
        req = types.SimpleNamespace(
            prompt=request_event.get_message_str(),
            system_prompt="",
            extra_user_content_parts=[],
        )

        await self.plugin.inject_history_on_request(request_event, req)

        self.assertEqual(req.extra_user_content_parts, [])

    async def test_botmesh_logical_group_whitelists_every_bound_bot(self):
        package_name = "astrbot_plugin_botmesh"
        integration_name = f"{package_name}.integration"
        previous_package = sys.modules.get(package_name)
        previous_integration = sys.modules.get(integration_name)
        package = types.ModuleType(package_name)
        package.__path__ = []
        integration = types.ModuleType(integration_name)

        def get_scope(*, umo, event=None):
            del umo, event
            return {
                "selector": "botmesh:main_group",
                "logical_group_id": "main_group",
                "selectors": [
                    "botmesh:main_group",
                    "A_GROUP",
                    "B_GROUP",
                    "onebot_main:GroupMessage:A_GROUP",
                    "onebot_second:GroupMessage:B_GROUP",
                ],
            }

        def normalize_message(*, umo, content, event=None):
            del umo, content, event
            return "已验证的 BotMesh 展示正文"

        integration.get_chat_history_scope = get_scope
        integration.normalize_chat_history_message = normalize_message
        sys.modules[package_name] = package
        sys.modules[integration_name] = integration

        def restore_modules():
            if previous_package is None:
                sys.modules.pop(package_name, None)
            else:
                sys.modules[package_name] = previous_package
            if previous_integration is None:
                sys.modules.pop(integration_name, None)
            else:
                sys.modules[integration_name] = previous_integration

        self.addCleanup(restore_modules)

        add_event = _Event(
            group_id="A_GROUP",
            umo="onebot_main:GroupMessage:A_GROUP",
            platform_id="onebot_main",
        )
        replies = [item async for item in self.plugin.historywatch_add(add_event)]
        self.assertEqual(self.config["listen_groups"], "botmesh:main_group")
        self.assertIn("botmesh:main_group", replies[0])

        other_bot_event = _Event(
            group_id="B_GROUP",
            umo="onebot_second:GroupMessage:B_GROUP",
            platform_id="onebot_second",
            text="带有隐藏传输帧的正文",
            message_id="mesh-1",
        )
        await self.plugin.capture_group_message(other_bot_event)
        records = self.plugin.store.query(
            umo=other_bot_event.unified_msg_origin,
            start_ts=0,
            end_ts=10**12,
        )
        self.assertEqual([record.content for record in records], ["已验证的 BotMesh 展示正文"])
        self.config["listen_groups"] = (
            "botmesh:main_group\nonebot_main:GroupMessage:A_GROUP"
        )
        _ = [
            item
            async for item in self.plugin.historywatch_remove(other_bot_event)
        ]
        self.assertEqual(self.config["listen_groups"], "")

    def test_capture_runs_before_botmesh_can_stop_the_event(self):
        priority = ChatHistoryContextPlugin.capture_group_message._filter_kwargs[
            "priority"
        ]
        self.assertGreater(priority, 120)


if __name__ == "__main__":
    unittest.main()
