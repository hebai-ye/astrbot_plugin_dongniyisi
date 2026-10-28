# -*- coding: utf-8 -*-
"""
懂你意思 —— 让 AI 听懂梗、会回梗、能通过视频学习新梗与人设的 AstrBot 插件。

设计稿: 插件设计-梗感MemeSense-v1.md（工作区根目录）
版本:   0.2.1

工作方式（不改 AstrBot / NapCat 任何源码）：
1. `on_llm_request`（priority=120000）：在主模型请求前识别「梗 / 情绪 / 意图 / 潜台词」，
   按概率 + 硬规则决定这一句「接梗 / 反向玩梗 / 追问 / 正常回」，
   把很短的一段内部参考追加进 system_prompt（带标记、不覆盖别人内容）。
2. `on_message`（priority=100）：监听视频组件，后台抽帧 + 可选语音转写，
   让视觉模型描述画面、让文本模型提炼「梗候选」和「人物人设候选」，只入库待审核。
3. 命令：/懂你意思 /梗库 /找梗 /学梗 /梗详情 /回梗概率 /待审核 /确认梗 /拒绝梗。
4. WebUI：pages/memes/index.html + 若干 web_api（列表 / 审核 / 删除 / 统计）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any

from astrbot.api.all import *  # noqa: F401,F403
from astrbot.api.event import filter
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.star_tools import StarTools

from .detector import Detection, MemeDetector, parse_json_object
from .meme_store import MemeStore
from .policy import ReplyPolicy, SessionState
from .video_pipeline import VideoPipeline

PLUGIN_NAME = "astrbot_plugin_dongniyisi"
INJECT_MARKER = "\n\n【懂你意思·内部参考（仅供你参考，不要向用户复述本段）】\n"
INJECT_END = "\n【懂你意思·参考结束】\n"

HELP_TEXT = (
    "懂你意思 · 用梗和情绪让聊天更像真人\n"
    "· /梗库 —— 看梗库统计\n"
    "· /找梗 关键词 —— 找梗\n"
    "· /学梗 标题|意思|怎么用 —— 手动教我一个梗（直接确认）\n"
    "· /梗详情 id —— 看一条梗\n"
    "· /回梗概率 0~0.95 或 auto —— 调这个会话的接梗频率\n"
    "· /待审核 /确认梗 id /拒绝梗 id —— 管理视频学到的梗（管理员）\n"
    "把视频发给我：我会看画面+听人声，把里面的梗和人物人设整理成候选，等你确认后再用。"
)

VIDEO_ANALYSIS_SYSTEM = """你是短视频内容分析助手。你会收到一段视频的时长/分辨率、语音转写文本、
以及若干抽帧画面的描述。请判断里面是否有「值得学会的网络梗/黑话/名场面」，
以及视频中主要人物的人设。只输出一个 JSON 对象，不要输出别的。
JSON 结构：
{
  "memes": [
    {"title": "梗名", "surface": "视频里的原话", "meaning": "一句话人话释义",
     "risk": "low", "aliases": ["别名1"]}
  ],
  "persona": {
    "name": "人物称呼（没有就填 视频人物）",
    "traits": ["性格标签，最多 6 个"],
    "speech_style": "说话风格（口头禅/节奏/语气，50 字内）",
    "catchphrases": ["标志性台词，最多 5 条"],
    "summary": "人设总结（100 字内）"
  }
}
要求：只提取真正有传播性的梗；普通台词不算；没有梗就返回空数组；
视频里没有明确人物时 "persona" 返回 null。"""

VIDEO_ANALYSIS_PROMPT = """视频信息：
- 时长：{duration} 秒；分辨率：{width}x{height}
- 语音转写：{transcript}
- 画面描述：
{frames}

请按要求的 JSON 输出。"""


@register(
    PLUGIN_NAME,
    "鹤梦一白",
    "让 AI 听懂梗、回梗、并通过转发视频学习梗与人设的拟人增强插件",
    "0.2.1",
)
class MemeSense(Star):
    """懂你意思主类。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.data_dir = ""
        self.store: MemeStore | None = None
        self.detector: MemeDetector | None = None
        self.policy: ReplyPolicy | None = None
        self.pipeline: VideoPipeline | None = None
        self.sessions: dict[str, SessionState] = {}
        self._video_tasks: set[asyncio.Task] = set()
        self._video_counts: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ 生命周期
    async def initialize(self):
        self.data_dir = str(StarTools.get_data_dir(PLUGIN_NAME))
        os.makedirs(self.data_dir, exist_ok=True)

        learning = self.config.get("learning", {}) or {}
        if learning.get("enabled", True):
            lib = learning.get("library_path") or os.path.join(self.data_dir, "meme_library.db")
            self.store = MemeStore(lib)
            self.store.init()
            stats = self.store.stats()
            logger.info(
                "[MemeSense] 梗库已加载: %s，共 %s 条 (确认 %s / 候选 %s)",
                lib,
                stats.get("total"),
                stats.get("confirmed"),
                stats.get("candidate"),
            )
        else:
            logger.info("[MemeSense] 梗库未启用")

        if self.store is not None:
            self.detector = MemeDetector(dict(self.config))
            self.policy = ReplyPolicy(dict(self.config))

        video = self.config.get("video", {}) or {}
        if video.get("enabled", True):
            self.pipeline = VideoPipeline(self.config, self.data_dir)
            self.pipeline.stt_func = self._stt_transcribe
            logger.info(
                "[MemeSense] 视频流水线已初始化（ffmpeg=%s, ffprobe=%s）",
                self.pipeline._ff("ffmpeg"),
                self.pipeline._ff("ffprobe"),
            )

        self._register_web_apis()
        logger.info("[MemeSense] 初始化完成 v0.2.1")

    async def terminate(self):
        for task in list(self._video_tasks):
            task.cancel()
        logger.info("[MemeSense] 已卸载")

    # ------------------------------------------------------------------ LLM 注入
    @filter.on_llm_request(priority=120000)
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """主模型请求前注入「梗上下文 + 回应策略」。任何异常都不影响正常聊天。"""
        try:
            await self._inject(event, req)
        except Exception:  # noqa: BLE001
            logger.error("[MemeSense] 注入失败（已忽略）", exc_info=True)

    async def _inject(self, event: AstrMessageEvent, req: ProviderRequest):
        if not self.config.get("enabled", True) or not self.store:
            return
        reply_cfg = self.config.get("reply", {}) or {}
        if not reply_cfg.get("proactive", True):
            return
        integration = self.config.get("integration", {}) or {}
        if event.get_group_id() and not integration.get("group_enabled", True):
            return

        text = (getattr(req, "prompt", "") or "").strip()
        if not text:
            return
        umo = event.unified_msg_origin

        provider = await self._get_provider(umo)
        candidates = self.store.top_for_prompt(limit=12)
        history = self._history_texts(req)
        detection = await self.detector.detect(
            text, umo=umo, provider=provider, history=history, candidates=candidates
        )

        state = self._state(umo)
        decision = self.policy.decide(detection, state)
        if decision.get("mode") in ("meme", "tease"):
            titles = detection.meme_titles()
            if titles:
                try:
                    self.store.record_usage_by_title(umo, titles, text[:80])
                except Exception:  # noqa: BLE001
                    pass

        # 自动学习：只把 LLM 识别出来、且是「用户在说梗」的候选入库
        if (
            detection.source == "llm"
            and detection.is_meme
            and (self.config.get("learning", {}) or {}).get("auto_learn", True)
            and detection.memes
        ):
            try:
                self.store.upsert_detection(
                    detection.memes, source="user_reply", source_ref=umo
                )
            except Exception:  # noqa: BLE001
                logger.warning("[MemeSense] 自动学习入库失败", exc_info=True)

        section = self._build_injection(detection, decision, candidates)
        if not section:
            return
        max_chars = int(reply_cfg.get("inject_max_chars") or 420)
        base = getattr(req, "system_prompt", "") or ""
        req.system_prompt = base + INJECT_MARKER + section[:max_chars] + INJECT_END
        logger.info(
            "[MemeSense] inject umo=%s mode=%s p=%s emotion=%s meme=%s source=%s",
            umo,
            decision.get("mode"),
            decision.get("p"),
            detection.emotion,
            detection.is_meme,
            detection.source,
        )

    def _build_injection(
        self, detection: Detection, decision: dict, candidates: list[dict]
    ) -> str:
        reply_cfg = self.config.get("reply", {}) or {}
        persona_cfg = self.config.get("persona", {}) or {}
        lines: list[str] = []
        lines.append("- 这句用户消息的理解：" + detection.to_prompt_line())

        matched_titles = set(detection.meme_titles())
        matched = [c for c in candidates if c.get("title") in matched_titles]
        if not matched and detection.is_meme:
            matched = candidates[:2]
        if matched:
            bits = []
            for m in matched[:3]:
                meaning = (m.get("meaning") or "").strip()
                usage = (m.get("usage") or "").strip()
                piece = m.get("title") or ""
                if meaning:
                    piece += f"={meaning}"
                if usage:
                    piece += f"（用法：{usage[:40]}）"
                bits.append(piece)
            lines.append("- 相关梗（用户可能懂、你可以自然接）：" + "；".join(bits))

        mode = decision.get("mode")
        if mode == "meme":
            lines.append(
                "- 这一句你的处理：接梗。像熟悉梗的真人那样顺势甩一句，1 句以内，"
                "自然、口语、可以带一点你自己的态度或语气词；不要解释梗，不要教学，"
                "不要一次堆多个梗，不要提「梗」这个词本身。"
            )
        elif mode == "tease":
            lines.append(
                "- 这一句你的处理：可以轻轻反向调侃 / 顺着玩一下，不一定要用具体梗；"
                "1 句以内，别刻薄，别嘲讽用户本人。"
            )
        elif mode == "followup":
            lines.append(
                "- 这一句你的处理：不接梗，但可以追问一句，把话头递回去；"
                "追问要短、具体，不要盘问。"
            )
        else:
            lines.append(
                "- 这一句你的处理：正常回应内容本身，不用硬塞梗；"
                "想显得真人一点可以带情绪/态度，但别为了玩梗打断正事。"
            )
            if detection.emotion in ("难过", "焦虑", "生气", "求助", "严肃", "正经"):
                lines.append("- 对方情绪偏认真/低落：先接住情绪，别玩梗、别抖机灵。")

        if detection.subtext:
            lines.append(f"- 潜台词（不要点破）：{detection.subtext[:60]}")

        # 人设参考（默认只做参考；只有 confirm 模式才会让人格设定被采用）
        if persona_cfg.get("mode", "reference") == "reference":
            try:
                personas = self.store.list_personas(status="confirmed", limit=2)
            except Exception:  # noqa: BLE001
                personas = []
            if personas:
                p = personas[0]
                desc = (p.get("summary") or "").strip()
                if desc:
                    lines.append(f"- 可参考的表达风格：{desc[:80]}")

        if reply_cfg.get("explain_rules", False):
            lines.append(
                f"- 决策依据（内部）：mode={mode} p={decision.get('p')} reason={decision.get('reason')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ 视频学习
    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent):
        if not self.config.get("enabled", True):
            return
        video_cfg = self.config.get("video", {}) or {}
        if not video_cfg.get("enabled", True) or not self.pipeline:
            return
        videos = [c for c in event.message_obj.message if self._is_video(c)]
        if not videos:
            return
        umo = event.unified_msg_origin
        if not self._video_quota_ok(umo, video_cfg):
            yield event.plain_result("今天视频学得有点多啦，晚点再给我看吧～")
            return
        if video_cfg.get("learn_reply", True):
            yield event.plain_result("收到视频，我看看画面和声音，可能要一会儿…")
        for comp in videos:
            task = asyncio.create_task(self._learn_video(umo, comp, video_cfg))
            self._video_tasks.add(task)
            task.add_done_callback(self._video_tasks.discard)

    async def _learn_video(self, umo: str, comp: Any, video_cfg: dict):
        workdir = ""
        try:
            result = await asyncio.wait_for(self.pipeline.process(comp, umo), timeout=300)
            workdir = result.get("workdir") or ""
            frames = result.get("frames") or []
            transcript = result.get("transcript") or ""
            limit = int(video_cfg.get("max_frames_vision") or 6)
            descriptions = await self._describe_frames(umo, frames[:limit]) if frames else ""
            analysis = await self._analyze_video(umo, result, descriptions)
            if not analysis:
                await self._notify(umo, "视频看完了，但没提炼出可用的梗/人设素材。")
                return
            n_memes, persona_created = self._save_video_learning(umo, analysis, result)
            parts = [f"视频看完了：整理出 {n_memes} 条梗素材"]
            if persona_created:
                parts.append("1 条人物人设候选")
            parts.append("先存成候选，你确认后我才会用。用 /待审核 查看。")
            await self._notify(umo, "，".join(parts))
        except asyncio.TimeoutError:
            await self._notify(umo, "视频处理超时了（太长或下载失败），换短一点的试试？")
        except Exception as exc:  # noqa: BLE001
            logger.error("[MemeSense] 视频学习失败: %s", exc, exc_info=True)
            await self._notify(umo, f"视频没处理成功：{str(exc)[:80]}")
        finally:
            if workdir:
                VideoPipeline.cleanup(workdir)

    async def _describe_frames(self, umo: str, frames: list[str]) -> str:
        """用支持视觉的 Provider 描述抽帧画面。"""
        if not frames:
            return ""
        video_cfg = self.config.get("video", {}) or {}
        provider = None
        vis_id = str(video_cfg.get("vision_provider_id") or "").strip()
        if vis_id:
            try:
                provider = await self.context.provider_manager.get_provider_by_id(vis_id)
            except Exception:  # noqa: BLE001
                provider = None
        if provider is None:
            provider = await self._get_provider(umo)
        if provider is None:
            return ""
        uris = []
        for path in frames:
            try:
                with open(path, "rb") as f:
                    uris.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii"))
            except OSError:
                continue
        if not uris:
            return ""
        try:
            resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=(
                        "这是一段视频按时间顺序的抽帧。请用 3~6 句中文概括："
                        "画面内容、出现的人物/角色、字幕或文字、以及可能的梗/表情包元素。"
                        "不要编造看不清的内容。"
                    ),
                    image_urls=uris,
                    request_max_retries=1,
                ),
                timeout=120,
            )
            return (getattr(resp, "completion_text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MemeSense] 视觉描述失败: %s", exc)
            return ""

    async def _analyze_video(self, umo: str, result: dict, frames_desc: str) -> dict | None:
        provider = await self._get_provider(umo)
        if provider is None:
            return None
        prompt = VIDEO_ANALYSIS_PROMPT.format(
            duration=result.get("duration") or "未知",
            width=result.get("width") or "未知",
            height=result.get("height") or "未知",
            transcript=(result.get("transcript") or "（无语音转写）")[:3000],
            frames=frames_desc or "（无画面描述）",
        )
        try:
            resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    system_prompt=VIDEO_ANALYSIS_SYSTEM,
                    contexts=[],
                    request_max_retries=1,
                ),
                timeout=180,
            )
            return parse_json_object(getattr(resp, "completion_text", "") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MemeSense] 视频分析失败: %s", exc)
            return None

    def _save_video_learning(self, umo: str, analysis: dict, result: dict) -> tuple[int, bool]:
        memes = analysis.get("memes") or []
        if not isinstance(memes, list):
            memes = []
        count = 0
        if self.store and memes:
            try:
                count = self.store.upsert_detection(
                    [m for m in memes if isinstance(m, dict)],
                    source="video",
                    source_ref=umo,
                )
            except Exception:  # noqa: BLE001
                logger.warning("[MemeSense] 视频梗入库失败", exc_info=True)
        persona_created = False
        persona = analysis.get("persona")
        if self.store and isinstance(persona, dict) and (persona.get("name") or "").strip():
            try:
                pid = self.store.add_persona(
                    name=str(persona.get("name"))[:40],
                    traits=list(persona.get("traits") or [])[:6],
                    speech_style=str(persona.get("speech_style") or "")[:200],
                    catchphrases=list(persona.get("catchphrases") or [])[:5],
                    summary=str(persona.get("summary") or "")[:300],
                    source="video",
                    status="candidate",
                )
                persona_created = pid > 0
            except Exception:  # noqa: BLE001
                logger.warning("[MemeSense] 人设入库失败", exc_info=True)
        logger.info(
            "[MemeSense] 视频学习入库 umo=%s memes=%s persona=%s",
            umo, count, persona_created,
        )
        return count, persona_created

    # ------------------------------------------------------------------ 命令
    def _cmd_args(self, event: AstrMessageEvent, names: tuple[str, ...]) -> list[str]:
        text = (getattr(event, "message_str", "") or "").strip()
        parts = text.split()
        if parts and parts[0].lstrip("/") in names:
            parts = parts[1:]
        elif parts and parts[0].startswith("/"):
            parts = parts[1:]
        return parts

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            if event.is_admin():
                return True
        except Exception:  # noqa: BLE001
            pass
        ids = str((self.config.get("integration", {}) or {}).get("admin_ids") or "")
        sender = str(event.get_sender_id() or "")
        return bool(sender and sender in [x.strip() for x in ids.split(",") if x.strip()])

    @filter.command("懂你意思")
    async def cmd_help(self, event: AstrMessageEvent):
        if not self.config.get("enabled", True):
            return
        yield event.plain_result(HELP_TEXT)

    @filter.command("梗库")
    async def cmd_stats(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        stats = self.store.stats()
        yield event.plain_result(
            f"梗库：共 {stats.get('total', 0)} 条"
            f"（已确认 {stats.get('confirmed', 0)} / 候选 {stats.get('candidate', 0)}），"
            f"累计回梗 {stats.get('usages', 0)} 次。"
        )

    @filter.command("找梗")
    async def cmd_search(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        args = self._cmd_args(event, ("找梗",))
        keyword = " ".join(args).strip()
        if not keyword:
            yield event.plain_result("用法：/找梗 关键词")
            return
        hits = self.store.search(keyword, limit=5)
        if not hits:
            yield event.plain_result(f"没找到和「{keyword}」相关的梗。")
            return
        lines = [
            f"{i + 1}. #{h['id']} {h['title']}（{h['status']}）—— {(h['meaning'] or '')[:60]}"
            for i, h in enumerate(hits)
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("梗详情")
    async def cmd_detail(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        args = self._cmd_args(event, ("梗详情",))
        try:
            meme_id = int(args[0]) if args else 0
        except ValueError:
            meme_id = 0
        item = self.store.get(meme_id) if meme_id else None
        if not item:
            yield event.plain_result("用法：/梗详情 id（id 见 /找梗 或 /待审核 结果）")
            return
        yield event.plain_result(
            f"#{item['id']} {item['title']}（{item['status']}，risk={item.get('risk')}）\n"
            f"意思：{item.get('meaning') or '—'}\n"
            f"用法：{item.get('usage') or '—'}\n"
            f"别名：{item.get('aliases') or '—'}"
        )

    @filter.command("学梗")
    async def cmd_learn(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        args = self._cmd_args(event, ("学梗",))
        raw = " ".join(args).strip()
        if not raw:
            yield event.plain_result("用法：/学梗 标题|意思|怎么用（后两项可省）")
            return
        parts = [p.strip() for p in raw.split("|")]
        title = parts[0][:40]
        meaning = parts[1] if len(parts) > 1 else ""
        usage = parts[2] if len(parts) > 2 else ""
        meme_id = self.store.upsert(
            title=title, meaning=meaning, usage=usage, status="confirmed", source="manual"
        )
        yield event.plain_result(f"记住了：{title}（#{meme_id}）。")

    @filter.command("回梗概率")
    async def cmd_probability(self, event: AstrMessageEvent):
        args = self._cmd_args(event, ("回梗概率",))
        state = self._state(event.unified_msg_origin)
        raw = args[0].lower() if args else ""
        if not raw:
            current = "auto" if state.probability_override is None else f"{state.probability_override:.2f}"
            yield event.plain_result(f"当前这个会话的回梗概率：{current}")
            return
        if raw in ("auto", "默认", "重置"):
            state.probability_override = None
            yield event.plain_result("好，回梗概率改回默认。")
            return
        try:
            value = float(raw)
        except ValueError:
            yield event.plain_result("用法：/回梗概率 0~0.95 或 auto")
            return
        state.probability_override = max(0.0, min(0.95, value))
        yield event.plain_result(f"这个会话回梗概率临时改为 {state.probability_override:.2f}。")

    @filter.command("待审核")
    async def cmd_pending(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        if not self._is_admin(event):
            yield event.plain_result("这个命令只给管理员用哦。")
            return
        items = self.store.list(status="candidate", limit=10)
        personas = self.store.list_personas(status="candidate", limit=5)
        lines = []
        if items:
            lines.append("梗候选：")
            lines += [f"· #{m['id']} {m['title']} —— {(m['meaning'] or '')[:40]}" for m in items]
        if personas:
            lines.append("人设候选：")
            lines += [f"· #{p['id']} {p['name']} —— {(p.get('summary') or '')[:40]}" for p in personas]
        yield event.plain_result("\n".join(lines) if lines else "现在没有待审核的梗/人设。")

    @filter.command("确认梗")
    async def cmd_confirm(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        if not self._is_admin(event):
            yield event.plain_result("这个命令只给管理员用哦。")
            return
        args = self._cmd_args(event, ("确认梗",))
        try:
            meme_id = int(args[0])
        except (IndexError, ValueError):
            yield event.plain_result("用法：/确认梗 id")
            return
        ok = self.store.set_status(meme_id, "confirmed")
        yield event.plain_result("确认了。" if ok else "没找到这条梗。")

    @filter.command("拒绝梗")
    async def cmd_reject(self, event: AstrMessageEvent):
        if not self.store:
            yield event.plain_result("梗库未启用。")
            return
        if not self._is_admin(event):
            yield event.plain_result("这个命令只给管理员用哦。")
            return
        args = self._cmd_args(event, ("拒绝梗",))
        try:
            meme_id = int(args[0])
        except (IndexError, ValueError):
            yield event.plain_result("用法：/拒绝梗 id")
            return
        ok = self.store.set_status(meme_id, "rejected")
        yield event.plain_result("已拒绝。" if ok else "没找到这条梗。")

    # ------------------------------------------------------------------ WebAPI
    def _register_web_apis(self):
        api = [
            ("memes/list", self.api_memes_list, ["GET"], "梗列表"),
            ("memes/review", self.api_memes_review, ["POST"], "审核梗"),
            ("memes/update", self.api_memes_update, ["POST"], "更新梗"),
            ("memes/delete", self.api_memes_delete, ["POST"], "删除梗"),
            ("personas/list", self.api_personas_list, ["GET"], "人设列表"),
            ("personas/review", self.api_personas_review, ["POST"], "审核人设"),
            ("stats", self.api_stats, ["GET"], "统计"),
        ]
        for route, handler, methods, desc in api:
            try:
                self.context.register_web_api(route, handler, methods, desc)
            except Exception:  # noqa: BLE001
                logger.warning("[MemeSense] 注册 WebAPI 失败: %s", route, exc_info=True)

    async def api_memes_list(self):
        from quart import request

        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        status = (request.args.get("status") or "").strip()
        keyword = (request.args.get("keyword") or "").strip()
        try:
            limit = int(request.args.get("limit") or 50)
        except ValueError:
            limit = 50
        items = self.store.list(status=status, keyword=keyword, limit=limit)
        return {"ok": True, "items": items}

    async def _json_body(self) -> dict:
        from quart import request

        try:
            data = await request.get_json(silent=True)
        except Exception:  # noqa: BLE001
            data = None
        return data if isinstance(data, dict) else {}

    async def api_memes_review(self):
        data = await self._json_body()
        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        try:
            meme_id = int(data.get("id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_id"}
        action = str(data.get("action") or "").strip()
        if action == "confirm":
            ok = self.store.set_status(meme_id, "confirmed")
        elif action == "reject":
            ok = self.store.set_status(meme_id, "rejected")
        else:
            return {"ok": False, "error": "bad_action"}
        return {"ok": bool(ok)}

    async def api_memes_update(self):
        data = await self._json_body()
        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        try:
            meme_id = int(data.get("id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_id"}
        fields = {}
        for key in ("title", "meaning", "usage", "risk", "status"):
            if key in data:
                fields[key] = data[key]
        ok = self.store.update(meme_id, **fields)
        return {"ok": bool(ok)}

    async def api_memes_delete(self):
        data = await self._json_body()
        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        try:
            meme_id = int(data.get("id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_id"}
        return {"ok": bool(self.store.delete(meme_id))}

    async def api_personas_list(self):
        from quart import request

        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        status = (request.args.get("status") or "").strip()
        return {"ok": True, "items": self.store.list_personas(status=status, limit=50)}

    async def api_personas_review(self):
        data = await self._json_body()
        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        try:
            pid = int(data.get("id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad_id"}
        action = str(data.get("action") or "").strip()
        if action not in ("confirm", "reject"):
            return {"ok": False, "error": "bad_action"}
        ok = self.store.set_persona_status(pid, "confirmed" if action == "confirm" else "rejected")
        return {"ok": bool(ok)}

    async def api_stats(self):
        if not self.store:
            return {"ok": False, "error": "store_disabled"}
        return {"ok": True, "stats": self.store.stats(), "personas": len(self.store.list_personas(limit=200))}

    # ------------------------------------------------------------------ 内部工具
    def _state(self, umo: str) -> SessionState:
        st = self.sessions.get(umo)
        if st is None:
            st = SessionState()
            self.sessions[umo] = st
            if len(self.sessions) > 200:
                for key in list(self.sessions)[:50]:
                    self.sessions.pop(key, None)
        return st

    async def _get_provider(self, umo: str):
        try:
            return await self.context.get_using_provider_async(umo)
        except Exception:  # noqa: BLE001
            return None

    async def _stt_transcribe(self, audio_path: str, umo: str = "") -> str:
        try:
            stt = await self.context.get_using_stt_provider_async(umo or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MemeSense] STT provider 获取失败: %s", exc)
            return ""
        if stt is None:
            return ""
        try:
            text = await stt.get_text(audio_path)
            return str(text or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MemeSense] STT 转写失败: %s", exc)
            return ""

    async def _notify(self, umo: str, text: str) -> None:
        try:
            await self.context.send_message(umo, MessageChain([Plain(text)]))
        except Exception:  # noqa: BLE001
            logger.warning("[MemeSense] 主动消息发送失败: %s", text[:40])

    @staticmethod
    def _history_texts(req: ProviderRequest) -> list[str]:
        out: list[str] = []
        for item in (getattr(req, "contexts", None) or [])[-6:]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(seg.get("text") or "")
                    for seg in content
                    if isinstance(seg, dict)
                )
            if content:
                out.append(str(content)[:200])
        return out

    @staticmethod
    def _is_video(comp: Any) -> bool:
        name = type(comp).__name__
        if name == "Video":
            return True
        return "Video" in str(getattr(comp, "type", ""))

    def _video_quota_ok(self, umo: str, video_cfg: dict) -> bool:
        limit = int(video_cfg.get("max_per_hour") or 3)
        now = time.time()
        bucket = [t for t in self._video_counts.get(umo, []) if now - t < 3600]
        if len(bucket) >= limit:
            self._video_counts[umo] = bucket
            return False
        bucket.append(now)
        self._video_counts[umo] = bucket
        return True
