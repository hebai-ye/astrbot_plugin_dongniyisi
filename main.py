# -*- coding: utf-8 -*-
"""
懂你意思 —— 让 AI 听懂梗、会回梗、能通过视频学习新梗与人设的 AstrBot 插件。

设计稿: 插件设计-梗感MemeSense-v1.md（工作区根目录）
版本:   0.3.3

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
from .stt_local import LocalSTT
from .video_pipeline import VideoPipeline
from .link_pipeline import LinkVideoPipeline, LinkVideoError

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

VIDEO_REACTION_SYSTEM = """你是一个有真实情绪、会开玩笑的聊天伙伴。用户刚给你发来一个视频，你已经看完了。
现在你要发出去的是你的「第一反应」，不是视频内容简介：
- 用第一人称、口语化，像在微信里随口说话；可以带一点语气词或 emoji（最多 1 个）
- 必须包含：你真实的感受/情绪（好笑、惊讶、无语、心疼、好奇……），以及你猜用户为什么发给你（求共鸣/求吐槽/分享/试探/想要安慰）
- 然后给出你的回应：吐槽、接梗、共情、追问或关心，自然合在一两句话里
- 绝对不要复述视频剧情、画面细节、台词；不要出现「视频里」「这个视频」「我看完了」这类播报式表达
- 不要解释你在分析，不要分点、不要引号、不要书名号
- 长度 20~70 字，最多两句
只输出你要发出去的那句话，不要任何前缀。"""

VIDEO_ANALYSIS_SYSTEM = """你是短视频内容分析助手。你会收到一段视频的时长/分辨率、语音转写文本、
以及若干抽帧画面的描述。请判断里面是否有「值得学会的网络梗/黑话/名场面」，
以及视频中主要人物的人设。只输出一个 JSON 对象，不要输出别的。
JSON 结构：
{
  "content": "一句话概括视频内容（40 字内，必须输出；画面和语音都看不出就写「内容不明确」）",
  "topics": ["视频主题/场景关键词，最多 5 个"],
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
要求：content 必须输出（它会被用来回答用户对视频内容的追问）；
只提取真正有传播性的梗，普通台词不算；没有梗就返回空数组；
视频里没有明确人物时 "persona" 返回 null。"""

VIDEO_ANALYSIS_PROMPT = """视频信息：
- 时长：{duration} 秒；分辨率：{width}x{height}
- 来源信息：{meta}
- 语音转写：{transcript}
- 画面描述：
{frames}

请按要求的 JSON 输出。"""


class _SimpleVideo:
    """把本地视频路径伪装成 AstrBot 视频组件（供 pipeline 复用）。"""

    type = "Video"

    def __init__(self, path: str, url: str = ""):
        self.file = ""
        self.path = path
        self.url = url
        self.cover = ""


@register(
    PLUGIN_NAME,
    "鹤梦一白",
    "让 AI 听懂梗、回梗、并通过转发视频学习梗与人设的拟人增强插件",
    "0.3.3",
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
        self.link: LinkVideoPipeline | None = None
        self.local_stt: LocalSTT | None = None
        self.sessions: dict[str, SessionState] = {}
        self._video_tasks: set[asyncio.Task] = set()
        self._video_counts: dict[str, list[float]] = {}
        self.video_memory: list[dict] = []
        self._video_memory_path = ""

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
        self._init_local_stt(video)
        if video.get("enabled", True):
            self.pipeline = VideoPipeline(self.config, self.data_dir)
            self.pipeline.stt_func = self._stt_transcribe
            logger.info(
                "[MemeSense] 视频流水线已初始化（ffmpeg=%s, ffprobe=%s）",
                self.pipeline._ff("ffmpeg"),
                self.pipeline._ff("ffprobe"),
            )
        link_cfg = self.config.get("link_video", {}) or {}
        if link_cfg.get("enabled", True):
            try:
                self.link = LinkVideoPipeline(link_cfg, self.data_dir)
                det = self.link._detect_ytdlp()
                logger.info("[MemeSense] 视频链接已启用（yt-dlp=%s, quality=%s, max_duration=%s）",
                            det, self.link.quality, self.link.max_duration)
            except Exception:  # noqa: BLE001
                logger.warning("[MemeSense] 视频链接初始化失败（忽略）", exc_info=True)

        self._video_memory_path = os.path.join(self.data_dir, "video_memory.json")
        self._load_video_memory()
        self._register_web_apis()
        logger.info("[MemeSense] 初始化完成 v0.3.3")

    async def terminate(self):
        for task in list(self._video_tasks):
            task.cancel()
        if self.local_stt is not None:
            try:
                await self.local_stt.close()
            except Exception:  # noqa: BLE001
                pass
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
        video_section = self._video_memory_section(umo)
        if video_section:
            section = (section + "\n" + video_section) if section else video_section
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
        links: list[str] = []
        link_cfg = self.config.get("link_video", {}) or {}
        if self.link and link_cfg.get("enabled", True):
            try:
                links = self.link.extract_urls(event.message_str or "")
            except Exception:  # noqa: BLE001
                links = []
        if not videos and not links:
            return
        umo = event.unified_msg_origin
        if not self._video_quota_ok(umo, video_cfg):
            yield event.plain_result("今天视频学得有点多啦，晚点再给我看吧～")
            return
        if video_cfg.get("learn_reply", True):
            ack = str(video_cfg.get("ack_text") or "等我看一下这个视频，马上回来～").strip()
            yield event.plain_result(ack)
        for comp in videos:
            task = asyncio.create_task(self._learn_video(umo, comp, video_cfg))
            self._video_tasks.add(task)
            task.add_done_callback(self._video_tasks.discard)
        for url in links[:2]:
            task = asyncio.create_task(self._learn_link(umo, url, video_cfg, link_cfg))
            self._video_tasks.add(task)
            task.add_done_callback(self._video_tasks.discard)

    async def _learn_video(self, umo: str, comp: Any, video_cfg: dict):
        workdir = ""
        try:
            result = await asyncio.wait_for(self.pipeline.process(comp, umo), timeout=300)
            workdir = result.get("workdir") or ""
            await self._finish_video(umo, result, video_cfg)
        except asyncio.TimeoutError:
            await self._notify(umo, "视频处理超时了（太长或下载失败），换短一点的试试？")
        except Exception as exc:  # noqa: BLE001
            logger.error("[MemeSense] 视频学习失败: %s", exc, exc_info=True)
            await self._notify(umo, f"视频没处理成功：{str(exc)[:80]}")
        finally:
            if workdir:
                VideoPipeline.cleanup(workdir)

    async def _learn_link(self, umo: str, url: str, video_cfg: dict, link_cfg: dict):
        """用户发来视频链接：解析 -> 下载 -> 走同一套抽帧/转写/分析/反应流程。"""
        workdir = ""
        try:
            prepared = await asyncio.wait_for(self.link.prepare(url, on_status=None),
                                             timeout=self.link.timeout + 120)
            workdir = prepared.get("workdir") or ""
            result = await asyncio.wait_for(
                self.pipeline.process(_SimpleVideo(prepared["path"], url), umo), timeout=300)
            result["workdir"] = ""  # 由本函数统一清理
            meta = prepared.get("meta") or {}
            result["meta"] = meta
            if prepared.get("subtitle"):
                result["subtitle"] = prepared["subtitle"]
            logger.info("[MemeSense] 链接视频已下载 umo=%s site=%s title=%s bytes=%s",
                        umo, meta.get("site"), str(meta.get("title"))[:40],
                        os.path.getsize(prepared["path"]) if os.path.isfile(prepared["path"]) else 0)
            await self._finish_video(umo, result, video_cfg)
        except asyncio.TimeoutError:
            await self._notify(umo, "这个视频链接处理超时了，换个短一点的试试？")
        except LinkVideoError as exc:
            await self._notify(umo, f"这个链接没看成功：{str(exc)[:120]}")
        except Exception as exc:  # noqa: BLE001
            logger.error("[MemeSense] 链接视频处理失败: %s", exc, exc_info=True)
            await self._notify(umo, f"这个链接没看成功：{str(exc)[:80]}")
        finally:
            if workdir:
                LinkVideoPipeline.cleanup(workdir)

    async def _finish_video(self, umo: str, result: dict, video_cfg: dict):
        """抽帧描述 -> 分析 -> 视频记忆 -> 学习入库 -> 生成第一反应。"""
        frames = result.get("frames") or []
        limit = int(video_cfg.get("max_frames_vision") or 6)
        descriptions = await self._describe_frames(umo, frames[:limit]) if frames else ""
        meta = result.get("meta") or {}
        analysis = await self._analyze_video(umo, result, descriptions, meta) or {}
        self._remember_video(umo, result, descriptions, analysis, meta)
        self._save_video_learning(umo, analysis, result)

        # 回复：默认给出「第一反应/感想」，而不是复述视频内容
        style = str(video_cfg.get("reply_style") or "reaction").strip().lower()
        reply = ""
        if style != "summary":
            reply = await self._generate_reaction(umo, result, analysis)
        if reply:
            logger.info("[MemeSense] 视频反应已生成 umo=%s chars=%s", umo, len(reply))
            await self._notify(umo, reply)
        else:
            # 回退：只给一句最小提示（不复述内容）
            await self._notify(umo, "这个视频我看了，不过一时没理清该说点啥……你发它是想让我陪你聊聊哪一点？")

    async def _describe_frames(self, umo: str, frames: list[str]) -> str:
        """用支持视觉的 Provider 描述抽帧画面。"""
        if not frames:
            return ""
        video_cfg = self.config.get("video", {}) or {}
        vis_id = str(video_cfg.get("vision_provider_id") or "").strip()
        chain: list[str] = []
        if vis_id:
            chain.append(vis_id)
        chain += ["deepseek-chat", "glm/glm-4v-flash"]
        providers = []
        seen = set()
        for pid in chain:
            if pid in seen:
                continue
            seen.add(pid)
            try:
                p = await self.context.provider_manager.get_provider_by_id(pid)
            except Exception:  # noqa: BLE001
                p = None
            if p is not None:
                providers.append((pid, p))
        cur = await self._get_provider(umo)
        if cur is not None:
            providers.append(("current", cur))
        if not providers:
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
        for pid, provider in providers:
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
                text = (getattr(resp, "completion_text", "") or "").strip()
                if text:
                    logger.info("[MemeSense] 视觉描述完成 provider=%s chars=%s", pid, len(text))
                    return text
                logger.warning("[MemeSense] 视觉描述为空 provider=%s，尝试下一个", pid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MemeSense] 视觉描述失败 provider=%s: %s", pid, exc)
        return ""

    async def _generate_reaction(self, umo: str, result: dict, analysis: dict) -> str:
        """看完视频后，生成带情绪、带用户意图推测的第一反应（不复述内容）。"""
        video_cfg = self.config.get("video", {}) or {}
        provider = await self._get_provider(umo)
        if provider is None:
            return ""
        content = str(analysis.get("content") or "").strip()
        topics = "、".join([str(t) for t in (analysis.get("topics") or [])][:5])
        memes = "、".join([str(m.get("title")) for m in (analysis.get("memes") or [])
                          if isinstance(m, dict) and m.get("title")][:5])
        persona_hint = str(video_cfg.get("persona_hint") or "").strip()
        facts = []
        if result.get("duration"):
            facts.append(f"时长约 {round(float(result['duration']))} 秒")
        if content:
            facts.append(f"内容概况：{content}")
        if topics:
            facts.append(f"主题：{topics}")
        meta = result.get("meta") or {}
        if meta.get("title"):
            facts.append(f"视频标题：{meta.get('title')}（作者：{meta.get('author') or '未知'}）")
        if meta.get("description"):
            facts.append("简介：" + str(meta.get("description"))[:300])
        if result.get("transcript"):
            facts.append("语音转写摘要：" + str(result.get("transcript"))[:400])
        if result.get("subtitle"):
            facts.append("字幕摘要：" + str(result.get("subtitle"))[:300])
        if result.get("frames"):
            facts.append("画面要点：" + str(result.get("frames"))[:500])
        if memes:
            facts.append(f"识别到的梗：{memes}")
        if persona_hint:
            facts.append(f"你的人设提示：{persona_hint}")
        prompt = "你刚看完用户发来的视频，这些是你看到/听到的事实（仅供你理解，不要在回复里复述）：\n" + "\n".join(facts)
        prompt += "\n\n现在发你的第一反应（感受+猜测用户为什么发+你的回应）。"
        try:
            resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=prompt,
                    system_prompt=VIDEO_REACTION_SYSTEM,
                    contexts=[],
                    request_max_retries=1,
                ),
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[MemeSense] 视频反应生成失败: %s", exc)
            return ""
        text = (getattr(resp, "completion_text", "") or "").strip()
        text = text.replace("\n", " ").replace("“", "").replace("”", "").replace("「", "").replace("」", "").strip()
        max_chars = int(video_cfg.get("reaction_max_chars") or 90)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip("，。；、 ") + "…"
        return text

    async def _analyze_video(self, umo: str, result: dict, frames_desc: str,
                             meta: dict | None = None) -> dict | None:
        provider = await self._get_provider(umo)
        if provider is None:
            return None
        meta = meta or result.get("meta") or {}
        meta_text = ""
        if meta:
            bits = []
            if meta.get("title"):
                bits.append(f"标题：{meta['title']}")
            if meta.get("author"):
                bits.append(f"UP主/作者：{meta['author']}")
            if meta.get("tags"):
                bits.append("标签：" + "、".join([str(t) for t in meta["tags"]][:8]))
            if meta.get("description"):
                bits.append(f"简介：{str(meta['description'])[:300]}")
            if meta.get("page_url"):
                bits.append(f"链接：{meta['page_url']}")
            meta_text = "；".join(bits) or "（无）"
        transcript = result.get("transcript") or ""
        subtitle = result.get("subtitle") or ""
        if not transcript and subtitle:
            transcript = "（视频字幕）" + str(subtitle)[:2000]
        prompt = VIDEO_ANALYSIS_PROMPT.format(
            duration=result.get("duration") or "未知",
            width=result.get("width") or "未知",
            height=result.get("height") or "未知",
            meta=meta_text or "（无）",
            transcript=(transcript or "（无语音转写）")[:3000],
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

    # ------------------------------------------------------------------ 视频短期记忆
    def _load_video_memory(self) -> None:
        try:
            if self._video_memory_path and os.path.isfile(self._video_memory_path):
                with open(self._video_memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.video_memory = [x for x in data if isinstance(x, dict)][-20:]
                logger.info("[MemeSense] 视频记忆已加载 %s 条", len(self.video_memory))
        except Exception:  # noqa: BLE001
            logger.warning("[MemeSense] 视频记忆加载失败（忽略）", exc_info=True)

    def _save_video_memory(self) -> None:
        try:
            if not self._video_memory_path:
                return
            tmp = self._video_memory_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.video_memory[-20:], f, ensure_ascii=False)
            os.replace(tmp, self._video_memory_path)
        except Exception:  # noqa: BLE001
            logger.warning("[MemeSense] 视频记忆写入失败（忽略）", exc_info=True)

    def _remember_video(self, umo: str, result: dict, descriptions: str,
                        analysis: dict, meta: dict | None = None) -> None:
        try:
            video_cfg = self.config.get("video", {}) or {}
            meta = meta or result.get("meta") or {}
            mem = {
                "ts": time.time(),
                "umo": umo,
                "title": str(meta.get("title") or "")[:80],
                "author": str(meta.get("author") or "")[:40],
                "url": str(meta.get("page_url") or "")[:200],
                "duration": result.get("duration"),
                "width": result.get("width"),
                "height": result.get("height"),
                "content": str(analysis.get("content") or "")[:120],
                "topics": list(analysis.get("topics") or [])[:6],
                "transcript": str(result.get("transcript") or "")[:1500],
                "frames": str(descriptions or "")[:1500],
                "memes": [
                    str(m.get("title") or "")[:30]
                    for m in (analysis.get("memes") or [])
                    if isinstance(m, dict)
                ][:6],
            }
            self.video_memory.append(mem)
            keep = int(video_cfg.get("memory_max") or 10)
            self.video_memory = self.video_memory[-max(1, keep):]
            self._save_video_memory()
            logger.info("[MemeSense] 视频记忆已存 umo=%s content=%s", umo, mem["content"][:40])
        except Exception:  # noqa: BLE001
            logger.warning("[MemeSense] 视频记忆保存失败（忽略）", exc_info=True)

    def _video_memory_section(self, umo: str) -> str:
        """最近一个视频的上下文，用于回答「视频里说了什么」这类追问。"""
        video_cfg = self.config.get("video", {}) or {}
        window = int(video_cfg.get("followup_window_sec") or 1800)
        if window <= 0:
            return ""
        now = time.time()
        mem = None
        for item in reversed(self.video_memory):
            if item.get("umo") == umo and now - float(item.get("ts") or 0) <= window:
                mem = item
                break
        if not mem:
            return ""
        minutes = max(1, int((now - float(mem.get("ts") or now)) / 60))
        title = str(mem.get("title") or "").strip()
        head = f"《{title}》" if title else "那个视频"
        lines = [f"- 你最近（约 {minutes} 分钟前）刚看完用户发的{head}：{mem.get('content') or '内容见下'}"]
        facts = []
        if mem.get("duration"):
            facts.append(f"时长约 {round(float(mem['duration']))} 秒")
        if mem.get("transcript"):
            facts.append("语音转写：" + str(mem["transcript"])[:220])
        if mem.get("frames"):
            facts.append("画面要点：" + str(mem["frames"])[:220])
        if mem.get("memes"):
            facts.append("识别到的梗：" + "、".join(mem["memes"][:5]))
        if facts:
            lines.append("- 视频事实（用户问起就基于这些回答，不要说你没收到视频）：" + "；".join(facts))
        lines.append("- 用户若问这个视频里有什么/让你总结，就基于以上内容回答；不要编造没看到的内容。")
        return "\n".join(lines)

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

    def _init_local_stt(self, video: dict) -> None:
        """按配置初始化本地 STT（Qwen3-ASR worker）。失败不影响其它功能。"""
        try:
            mode = str(video.get("stt_mode") or "auto").strip().lower()
            if mode in ("off", "provider", "disabled"):
                logger.info("[MemeSense] 本地 STT 未启用（stt_mode=%s）", mode)
                return
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            python_exe = str(
                video.get("stt_python")
                or os.path.join(os.path.dirname(self.data_dir), "..", "..", "..", "python", "python.exe")
            ).strip()
            python_exe = os.path.normpath(python_exe) if python_exe else ""
            model_dir = str(video.get("stt_model_dir") or "").strip()
            worker = str(video.get("stt_worker") or os.path.join(plugin_dir, "scripts", "stt_worker.py"))
            local = LocalSTT(
                python_exe=python_exe,
                worker_script=worker,
                model_dir=model_dir,
                num_threads=int(video.get("stt_num_threads") or 4),
                max_new_tokens=int(video.get("stt_max_new_tokens") or 512),
                hotwords=str(video.get("stt_hotwords") or ""),
                timeout_scale=float(video.get("stt_timeout_scale") or 6.0),
            )
            if not local.available():
                logger.warning(
                    "[MemeSense] 本地 STT 配置不完整，跳过（python=%s worker=%s model=%s）",
                    python_exe,
                    worker,
                    model_dir,
                )
                return
            self.local_stt = local
            logger.info(
                "[MemeSense] 本地 STT 已配置：model=%s threads=%s（首次转写时拉起 worker）",
                model_dir,
                local.num_threads,
            )
        except Exception:  # noqa: BLE001
            logger.warning("[MemeSense] 本地 STT 初始化失败（忽略）", exc_info=True)

    async def _stt_transcribe(self, audio_path: str, umo: str = "") -> str:
        """优先本地 STT，失败/未配置时回退 AstrBot STT Provider。"""
        video = self.config.get("video", {}) or {}
        mode = str(video.get("stt_mode") or "auto").strip().lower()
        if self.local_stt is not None and mode != "provider":
            try:
                timeout = max(60.0, float(video.get("stt_timeout_sec") or 600))
                text = await self.local_stt.transcribe(audio_path, timeout=timeout)
                logger.info(
                    "[MemeSense] 本地 STT 完成：%s 字（%s）",
                    len(text or ""),
                    mode,
                )
                return text
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MemeSense] 本地 STT 失败，回退 Provider: %s", exc)
        if mode == "local":
            return ""
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
