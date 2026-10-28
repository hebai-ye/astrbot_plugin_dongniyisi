# -*- coding: utf-8 -*-
"""懂你意思 · 识别引擎

职责：把「用户这句话」变成结构化信号：
    - 是否在用梗（is_meme）+ 具体梗（title/surface/meaning/risk）
    - 情绪（emotion）、对话基调（tone）、意图（intent）、潜台词（subtext）
    - 给主模型的建议（reply_strategy：should_meme / mode / reason）

实现要点：
    1. 主路径：调用 AstrBot 当前会话的聊天 Provider，用严格的 JSON 提示词拿结构化结果；
    2. 兜底路径：LLM 超时/失败/超频时用启发式（关键词 + 标点 + 梗库关键词匹配），保证插件永不阻塞对话；
    3. 缓存：同一会话同一句话 5 分钟内复用结果，避免重复烧 token；
    4. 限流：每分钟最多 N 次 LLM 判断（默认 20），超过后走启发式。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

SYSTEM_PROMPT = """你是一个中文网络语言的「语义 + 情绪 + 梗」分析引擎。
你的任务不是聊天，而是输出结构化 JSON，供另一个 AI 参考。

输出要求：
1. 只输出一个 JSON 对象；不要 markdown 代码块，不要解释，不要多余文字。
2. 字段固定如下（缺省也要出现）：
{
  "is_meme": false,
  "memes": [
    {"title": "梗的名字", "surface": "用户原句里的说法", "meaning": "一句话人话释义", "risk": "low"}
  ],
  "emotion": "中性",
  "tone": "日常",
  "intent": "陈述",
  "subtext": "",
  "reply_strategy": {"should_meme": 0.0, "mode": "normal", "reason": "一句简短理由"}
}
3. 枚举取值：
   emotion: 轻松 / 开心 / 难过 / 焦虑 / 生气 / 撒娇 / 暧昧 / 严肃 / 中性
   tone: 日常 / 玩闹 / 吐槽 / 正经 / 亲密
   intent: 陈述 / 提问 / 吐槽 / 撒娇 / 调侃AI / 试探 / 求助 / 分享
   reply_strategy.mode: meme（接梗）/ tease（反向调侃但不用梗）/ normal（正常回）/ followup（追问或接话）
4. 判断要克制：只有真的是「梗 / 黑话 / 抽象表达 / 二创梗」才 is_meme=true；普通口语、语气词、口头禅不算梗。
5. 优先使用「本地梗库候选」里的名字；本地没有的可以自己给出名字。
6. 高风险梗（辱骂、地域歧视、政治敏感、低俗色情）risk 标 high，meaning 只写中性释义，subtext 不要展开。
7. should_meme 表示「这次适合接梗的程度」：用户认真说事/求助/难过时给 0~0.15；用户在玩闹、吐槽、互相抛梗时给 0.55~0.95。
"""

LEARN_SYSTEM_PROMPT = """你是一个「梗与人设」提炼引擎。你会收到一段视频的转写文本和若干抽帧描述。
请提炼其中的网络梗与人物人设素材，只输出一个 JSON 对象：

{
  "summary": "这段视频讲了什么（80 字内）",
  "memes": [
    {"title": "梗名", "surface": "视频里的原话", "meaning": "一句话人话释义", "risk": "low", "aliases": ["别名1"]}
  ],
  "persona": {
    "name": "视频中主要人物/角色的称呼（没有就填 视频人物）",
    "traits": ["性格标签，最多 6 个"],
    "speech_style": "说话风格（口头禅/节奏/语气，50 字内）",
    "catchphrases": ["标志性台词，最多 5 条"],
    "summary": "人设总结（100 字内）"
  }
}
要求：梗只提取真正有传播性的网络梗/黑话/名场面；普通台词不算。没有梗就返回空数组。persona 字段如果视频里没有明确人物，返回 null。
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)

SERIOUS_HINTS = (
    "难过", "伤心", "焦虑", "求助", "救命", "怎么办", "生气", "烦", "委屈",
    "哭", "害怕", "担心", "压力", "失眠", "受不了", "崩溃", "想死",
)
PLAYFUL_HINTS = (
    "哈哈", "笑死", "草", "乐", "绝了", "离谱", "绷不住", "典", "乐子",
    "嘛", "捏", "hhh", "233", "xswl", "awsl", "yyds", "6", "？？？", "???",
)
QUESTION_HINTS = ("?", "？", "吗", "呢", "什么", "怎么", "为啥", "为什么", "哪", "谁", "如何")
MEME_MARKERS = (
    "绷不住", "典中典", "就这", "急了", "急了急了", "乐", "草", "hhh", "麻了",
    "破防", "退退退", "润", "摆烂", "躺平", "笑不活了", "整活", "抽象", "唐",
    "2333", "xswl", "awsl", "yyds", "u1s1", "有一说一", "典", "栓Q", "谢谢啊",
)

EMOTION_FACTORS = {
    "轻松": 1.30,
    "开心": 1.20,
    "玩闹": 1.40,
    "吐槽": 1.30,
    "撒娇": 1.10,
    "暧昧": 1.05,
    "中性": 1.00,
    "正经": 0.50,
    "严肃": 0.15,
    "难过": 0.0,
    "焦虑": 0.0,
    "生气": 0.0,
    "求助": 0.0,
}

ALLOWED_MODES = ("meme", "tease", "normal", "followup")


@dataclass
class Detection:
    """一次识别的结果。"""

    is_meme: bool = False
    memes: list[dict] = field(default_factory=list)
    emotion: str = "中性"
    tone: str = "日常"
    intent: str = "陈述"
    subtext: str = ""
    should_meme: float = 0.0
    suggest_mode: str = "normal"
    suggest_reason: str = ""
    source: str = "heuristic"  # llm / heuristic / cache
    raw: dict = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    def emotion_factor(self) -> float:
        return EMOTION_FACTORS.get(self.emotion, 1.0)

    def meme_titles(self) -> list[str]:
        out = []
        for m in self.memes:
            title = str(m.get("title") or "").strip()
            if title:
                out.append(title)
        return out

    def to_prompt_line(self) -> str:
        """给主模型看的简短摘要（不含原始 JSON）。"""
        bits = [f"情绪={self.emotion}", f"基调={self.tone}", f"意图={self.intent}"]
        if self.is_meme:
            names = "、".join(self.meme_titles()[:2]) or "（未命名）"
            bits.append(f"检测到梗={names}")
        if self.subtext:
            bits.append(f"潜台词={self.subtext[:40]}")
        return "；".join(bits)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def parse_json_object(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。"""
    if not text:
        return None
    cleaned = _CODE_FENCE_RE.sub("", text.strip())
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        return None
    for candidate in (match.group(0), cleaned):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


class MemeDetector:
    """梗 / 情绪 / 意图识别器。"""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("detector", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.cache_seconds = float(cfg.get("cache_seconds", 300))
        self.rate_limit_per_min = int(cfg.get("rate_limit_per_min", 20))
        self.timeout_sec = float(cfg.get("timeout_sec", 8))
        self.max_tokens = int(cfg.get("max_tokens", 420))
        self.max_candidates = int(cfg.get("max_candidates", 12))
        self._cache: dict[str, tuple[float, Detection]] = {}
        self._calls: deque[float] = deque()

    # ------------------------------------------------------------------ public
    def heuristic(self, text: str, library_hits: list[dict] | None = None) -> Detection:
        """离线启发式：没有 LLM 时也能大致判断。"""
        text = text or ""
        lower = text.lower()
        serious = any(h in text for h in SERIOUS_HINTS)
        playful = any(h in lower for h in PLAYFUL_HINTS)
        is_question = any(h in text for h in QUESTION_HINTS)
        marker = next((m for m in MEME_MARKERS if m.lower() in lower), "")

        emotion = "中性"
        tone = "日常"
        should = 0.0
        mode = "normal"
        if serious:
            emotion = "正经"
            tone = "正经"
            should = 0.05
        elif playful or marker:
            emotion = "轻松"
            tone = "玩闹"
            should = 0.55 if marker else 0.35
            mode = "meme" if should >= 0.5 else "normal"
        elif is_question:
            emotion = "中性"
            tone = "日常"
            should = 0.15
            mode = "normal"

        memes: list[dict] = []
        for hit in library_hits or []:
            title = str(hit.get("title") or "")
            if title and title in text:
                memes.append(
                    {
                        "title": title,
                        "surface": title,
                        "meaning": str(hit.get("meaning") or "")[:40],
                        "risk": str(hit.get("risk") or "low"),
                    }
                )
        if not memes and marker:
            memes.append({"title": marker, "surface": marker, "meaning": "", "risk": "low"})

        return Detection(
            is_meme=bool(memes),
            memes=memes[:3],
            emotion=emotion,
            tone=tone,
            intent="提问" if is_question else ("吐槽" if playful else "陈述"),
            subtext="",
            should_meme=should,
            suggest_mode=mode,
            suggest_reason="heuristic",
            source="heuristic",
        )

    async def detect(
        self,
        text: str,
        umo: str,
        provider: Any,
        history: list[str] | None = None,
        candidates: list[dict] | None = None,
    ) -> Detection:
        """识别一句话。失败时返回启发式结果，绝不抛异常。"""
        text = (text or "").strip()
        if not text:
            return Detection(source="heuristic")

        key = hashlib.sha1(f"{umo}|{text}".encode("utf-8")).hexdigest()
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < self.cache_seconds:
            det = cached[1]
            det.source = "cache"
            return det

        library_hits = self._match_library(text, candidates or [])
        fallback = self.heuristic(text, library_hits)

        if not self.enabled or provider is None or not self._allow_llm():
            return fallback

        try:
            prompt = self._build_prompt(text, history, candidates or [], library_hits)
            resp = await _with_timeout(
                provider.text_chat(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    contexts=[],
                    request_max_retries=1,
                ),
                self.timeout_sec,
            )
            raw_text = getattr(resp, "completion_text", "") or ""
            data = parse_json_object(raw_text)
            if not data:
                return fallback
            det = self._from_data(data, fallback)
            self._remember(key, det)
            return det
        except Exception:  # noqa: BLE001 - 识别失败不允许影响聊天
            return fallback

    # ------------------------------------------------------------------ internals
    def _allow_llm(self) -> bool:
        now = time.time()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= max(1, self.rate_limit_per_min):
            return False
        self._calls.append(now)
        return True

    def _remember(self, key: str, det: Detection) -> None:
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = (time.time(), det)

    @staticmethod
    def _match_library(text: str, candidates: list[dict]) -> list[dict]:
        hits = []
        for item in candidates:
            title = str(item.get("title") or "")
            if not title:
                continue
            if title in text:
                hits.append(item)
                continue
            aliases = item.get("aliases") or []
            if isinstance(aliases, str):
                try:
                    aliases = json.loads(aliases)
                except json.JSONDecodeError:
                    aliases = []
            if any(str(a) and str(a) in text for a in aliases):
                hits.append(item)
        return hits[:3]

    def _build_prompt(
        self,
        text: str,
        history: list[str] | None,
        candidates: list[dict],
        library_hits: list[dict],
    ) -> str:
        parts = []

        if history:
            recent = [h for h in history[-4:] if h]
            if recent:
                parts.append("【最近对话】\n" + "\n".join(f"- {h[:120]}" for h in recent))

        if candidates:
            rows = []
            for item in candidates[: self.max_candidates]:
                title = str(item.get("title") or "")[:24]
                if not title:
                    continue
                aliases = item.get("aliases") or []
                if isinstance(aliases, str):
                    try:
                        aliases = json.loads(aliases)
                    except json.JSONDecodeError:
                        aliases = []
                alias_text = "/".join(str(a) for a in aliases[:3])
                meaning = str(item.get("meaning") or "")[:60]
                rows.append(f"- {title} | {alias_text} | {meaning}")
            if rows:
                parts.append("【本地梗库候选】\n" + "\n".join(rows))

        if library_hits:
            parts.append(
                "【关键词直接命中】"
                + "、".join(str(h.get("title")) for h in library_hits)
            )

        parts.append(f"【用户这条消息】\n{text[:600]}")
        parts.append("请输出 JSON。")
        return "\n\n".join(parts)

    @staticmethod
    def _from_data(data: dict, fallback: Detection) -> Detection:
        memes_raw = data.get("memes") or []
        memes: list[dict] = []
        if isinstance(memes_raw, list):
            for item in memes_raw[:3]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                risk = str(item.get("risk") or "low").lower()
                if risk not in ("low", "mid", "high"):
                    risk = "low"
                memes.append(
                    {
                        "title": title[:40],
                        "surface": str(item.get("surface") or "")[:80],
                        "meaning": str(item.get("meaning") or "")[:160],
                        "risk": risk,
                    }
                )

        strategy = data.get("reply_strategy") or {}
        if not isinstance(strategy, dict):
            strategy = {}
        mode = str(strategy.get("mode") or fallback.suggest_mode).lower()
        if mode not in ALLOWED_MODES:
            mode = "normal"

        emotion = str(data.get("emotion") or "中性").strip()[:8] or "中性"
        return Detection(
            is_meme=bool(data.get("is_meme")) or bool(memes),
            memes=memes,
            emotion=emotion,
            tone=str(data.get("tone") or "日常")[:8],
            intent=str(data.get("intent") or "陈述")[:8],
            subtext=str(data.get("subtext") or "")[:100],
            should_meme=_clamp01(strategy.get("should_meme"), fallback.should_meme),
            suggest_mode=mode,
            suggest_reason=str(strategy.get("reason") or "")[:120],
            source="llm",
            raw=data if isinstance(data, dict) else {},
        )


async def _with_timeout(coro, timeout: float):
    """给 awaiter 加超时（不依赖 asyncio.wait_for 的兼容问题）。"""
    import asyncio

    return await asyncio.wait_for(coro, timeout=max(1.0, timeout))
