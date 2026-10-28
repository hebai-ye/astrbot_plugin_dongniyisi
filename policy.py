# -*- coding: utf-8 -*-
"""懂你意思 · 回梗决策器

输入：Detection（梗/情绪/意图）+ 会话状态
输出：这一次「怎么回」
    - mode: meme（接梗）/ tease（反向调侃、不用梗）/ followup（追问接话）/ normal（正常回）
    - probability / reason（写日志、给调试用）

设计原则（为什么需要这些规则）：
    1. 真人不会每条都接梗：需要基础概率 + 情绪修正 + 最近回梗次数限流；
    2. 真人不会把同一个梗连用好几遍：同梗短时间重复要衰减；
    3. 认真/求助/难过时不接梗：硬规则优先于概率；
    4. 刚回完梗要「收一下」：冷却期内概率打折，避免 AI 感；
    5. 用户明确要求「活跃/冷静」时会话级覆盖优先。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """一个会话（umo）的运行状态。"""

    recent: list[dict] = field(default_factory=list)  # 最近决策记录
    last_meme_at: float = 0.0
    recent_meme_titles: list[tuple[float, str]] = field(default_factory=list)
    tone_override: str = ""  # active / cool / ""
    override_until: float = 0.0
    probability_override: float | None = None

    def is_override_active(self) -> bool:
        if self.tone_override and self.override_until and time.time() > self.override_until:
            self.tone_override = ""
            self.override_until = 0.0
        return time.time() < self.override_until

    def add(self, record: dict) -> None:
        self.recent.append(record)
        self.recent = self.recent[-50:]

    def recalled_in_last(self, n: int) -> int:
        return sum(1 for r in self.recent[-n:] if r.get("mode") in ("meme", "tease"))

    def count_recent_memes(self, window: int = 10) -> int:
        return sum(1 for r in self.recent[-window:] if r.get("mode") == "meme")


class ReplyPolicy:
    """根据 Detection 与会话状态决定回梗方式。"""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("reply", {}) or {}
        self.base_probability = float(cfg.get("base_probability", 0.30))
        self.max_per_10 = int(cfg.get("max_per_10", 2))
        self.cooldown_sec = float(cfg.get("cooldown_sec", 90))
        self.serious_emotions = set(cfg.get("serious_emotions") or [
            "难过", "焦虑", "生气", "严肃", "正经", "求助",
        ])
        self.block_risky = bool(cfg.get("block_risky", True))
        self.repeat_decay = float(cfg.get("repeat_decay", 0.45))
        self.tease_extra = float(cfg.get("tease_extra", 0.12))
        self.followup_prob = float(cfg.get("followup_prob", 0.25))

    # ------------------------------------------------------------------ public
    def decide(self, detection: Any, state: SessionState) -> dict:
        """返回 {"mode", "probability", "reason", "forced"}。"""
        base = float(state.probability_override) if state.probability_override is not None else self.base_probability
        emotion = getattr(detection, "emotion", "中性")
        is_meme = bool(getattr(detection, "is_meme", False))
        intent = str(getattr(detection, "intent", "陈述"))
        should = float(getattr(detection, "should_meme", 0.0) or 0.0)
        suggest_mode = str(getattr(detection, "suggest_mode", "normal") or "normal")
        titles = list(getattr(detection, "meme_titles", lambda: [])())
        now = time.time()

        # 硬规则 0：会话级「冷静」覆盖
        if state.is_override_active() and state.tone_override == "cool":
            return self._decide(state, "normal", 0.0, "session_cool", now)

        # 硬规则 1：严肃情绪不接梗
        if emotion in self.serious_emotions:
            return self._decide(state, "normal", 0.0, f"serious_emotion:{emotion}", now)

        # 硬规则 2：10 条内回梗次数到顶
        if state.count_recent_memes(10) >= max(0, self.max_per_10):
            return self._decide(state, "normal", 0.0, "quota_10", now)

        # 概率合成
        p = base * detection.emotion_factor()
        p *= 0.55 + 0.9 * min(1.0, should)
        if is_meme:
            p *= 1.25
        if intent in ("求助", "提问") and not is_meme:
            p *= 0.6
        if state.tone_override and state.is_override_active() and state.tone_override == "active":
            p *= 1.35
        # 冷却：刚接过梗 → 打折
        if state.last_meme_at and now - state.last_meme_at < self.cooldown_sec:
            p *= 0.5
        # 同梗重复衰减
        recent_titles = [t for ts, t in state.recent_meme_titles if now - ts < 600]
        if any(t in titles for t in recent_titles if t):
            p *= self.repeat_decay
        # 高风险梗：只理解不模仿
        if self.block_risky and any(str(m.get("risk")) == "high" for m in getattr(detection, "memes", [])):
            p = 0.0
        p = max(0.0, min(0.95, p))

        roll = random.random()
        if is_meme and roll < p:
            mode = "meme"
        elif roll < p + self.tease_extra * (0.5 + should):
            mode = "tease"
        elif suggest_mode == "followup" and random.random() < self.followup_prob:
            mode = "followup"
        else:
            mode = "normal"
        return self._decide(state, mode, p, f"roll={roll:.2f}<p={p:.2f}", now)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _decide(state: SessionState, mode: str, probability: float, reason: str, now: float) -> dict:
        record = {
            "mode": mode,
            "p": round(probability, 3),
            "reason": reason,
            "ts": now,
        }
        state.add(record)
        if mode in ("meme", "tease"):
            state.last_meme_at = now
        return record
