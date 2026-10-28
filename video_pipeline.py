# -*- coding: utf-8 -*-
"""懂你意思 · 视频流水线

把「用户转发的视频」变成可供模型分析的素材：
    视频来源 → 本地文件 → ffprobe 元信息 → ffmpeg 抽帧（jpg）
             → ffmpeg 抽音轨（16k 单声道 wav）→ STT 转写（可选）

只负责「素材化」，不调用 LLM；LLM 提炼梗与人设在 main.py / detector.py 里做。

ffmpeg 查找顺序：配置 ffmpeg_path → 环境变量 FFMPEG_PATH → PATH 里的 ffmpeg/ffprobe
支持 Windows 便携版（把 ffmpeg.exe 放 D:\\DeepSeekQQ\\tools\\ffmpeg\\bin 后填配置即可）。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from typing import Any, Awaitable, Callable


class VideoPipeline:
    def __init__(self, config: dict | None = None, data_dir: str = ""):
        cfg = (config or {}).get("video", {}) or {}
        self.cfg = cfg
        self.data_dir = data_dir
        self.ffmpeg_path = str(cfg.get("ffmpeg_path") or "").strip()
        self.ffprobe_path = str(cfg.get("ffprobe_path") or "").strip()
        self.keep_video = bool(cfg.get("keep_video", False))
        self.max_mb = int(cfg.get("max_mb", 100))
        self.max_duration_sec = int(cfg.get("max_duration_sec", 600))
        self.frame_interval_sec = max(1, int(cfg.get("frame_interval_sec", 4)))
        self.max_frames = max(1, min(24, int(cfg.get("max_frames", 8))))
        self.stt_enabled = bool(cfg.get("stt_enabled", True))
        # 由 main.py 注入：async def stt(audio_path: str, umo: str) -> str
        self.stt_func: Callable[[str, str], Awaitable[str]] | None = None

    # ------------------------------------------------------------------ public
    async def process(self, comp: Any, umo: str = "") -> dict:
        """返回 {source, duration, width, height, frames, audio, transcript, workdir}。"""
        src = await self._resolve_source(comp)
        workdir = tempfile.mkdtemp(prefix="dongniyisi_")
        try:
            meta = await self._probe(src)
            frames = await self._extract_frames(src, workdir)
            audio = await self._extract_audio(src, workdir)
            transcript = ""
            if audio and self.stt_enabled and self.stt_func is not None:
                try:
                    transcript = await asyncio.wait_for(self.stt_func(audio, umo), timeout=240)
                except Exception as exc:  # noqa: BLE001 - STT 失败不阻断
                    self._log(f"STT 失败: {exc}")
                    transcript = ""
            return {
                "source": src,
                "duration": meta.get("duration"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "frames": frames,
                "audio": audio,
                "transcript": transcript or "",
                "workdir": workdir,
            }
        except Exception:
            self.cleanup(workdir)
            raise

    @staticmethod
    def cleanup(workdir: str) -> None:
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------ source
    async def _resolve_source(self, comp: Any) -> str:
        """把视频组件解析成本地文件路径。"""
        candidates = []
        for attr in ("path", "file", "url"):
            value = getattr(comp, attr, None)
            if value:
                candidates.append(str(value))

        for value in candidates:
            local = self._to_local_path(value)
            if local:
                return local

        for value in candidates:
            if value.startswith(("http://", "https://")) and (
                "multimedia" in value or value.endswith((".mp4", ".mov", ".m4v", ".flv"))
            ):
                return await self._download(value)

        raise ValueError(
            "无法拿到视频文件（QQ 视频可能是 file id，需要 NapCat 开启文件下载或改用链接）。"
            f" 组件信息: {candidates[:3]}"
        )

    @staticmethod
    def _to_local_path(value: str) -> str | None:
        if value.startswith("file://"):
            value = value[7:]
        if os.path.isfile(value):
            return value
        return None

    async def _download(self, url: str) -> str:
        import aiohttp

        cache = os.path.join(self.data_dir, "video_cache") or "video_cache"
        os.makedirs(cache, exist_ok=True)
        out = os.path.join(cache, f"video_{int(time.time())}.mp4")
        max_bytes = self.max_mb * 1024 * 1024
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=180) as resp:
                resp.raise_for_status()
                size = 0
                with open(out, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1 << 20):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"视频超过 {self.max_mb}MB 上限")
                        f.write(chunk)
        return out

    # ------------------------------------------------------------------ ffmpeg
    def _ff(self, name: str = "ffmpeg") -> str:
        """查找 ffmpeg/ffprobe 可执行文件。

        优先级：显式配置 → 环境变量 → 从另一个可执行文件所在目录推导 →
        常见便携目录 → PATH 里的名字。ffmpeg 与 ffprobe 通常同目录，
        因此只要配置了 ffmpeg_path，ffprobe 就能自动推导出来。
        """
        configured = self.ffprobe_path if name == "ffprobe" else self.ffmpeg_path
        if configured and os.path.isfile(configured):
            return configured

        for env_name in (name.upper() + "_PATH", name.upper()):
            env = os.environ.get(env_name)
            if env and os.path.isfile(env):
                return env

        # 从已配置的另一个可执行文件推导（ffmpeg 配置了，ffprobe 就在同目录）
        sibling_base = self.ffmpeg_path if name == "ffprobe" else self.ffprobe_path
        if sibling_base:
            base_dir = os.path.dirname(sibling_base)
            for ext in ("", ".exe"):
                candidate = os.path.join(base_dir, name + ext)
                if os.path.isfile(candidate):
                    return candidate

        # 常见便携目录
        for base in (
            os.path.join(self.data_dir, "tools", "ffmpeg", "bin") if self.data_dir else "",
            r"D:\DeepSeekQQ\tools\ffmpeg\bin",
        ):
            if not base:
                continue
            for ext in ("", ".exe"):
                candidate = os.path.join(base, name + ext)
                if os.path.isfile(candidate):
                    return candidate

        # 最后交给系统 PATH
        return name

    async def _probe(self, src: str) -> dict:
        proc = await asyncio.create_subprocess_exec(
            self._ff("ffprobe"),
            "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", src,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("ffprobe 读取视频失败（检查 ffmpeg 配置）")
        try:
            info = json.loads(stdout or b"{}")
        except json.JSONDecodeError:
            info = {}
        vstream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        duration = float(info.get("format", {}).get("duration") or 0)
        if duration and duration > self.max_duration_sec:
            raise ValueError(f"视频时长 {duration:.0f}s 超过上限 {self.max_duration_sec}s")
        return {
            "duration": duration,
            "width": vstream.get("width"),
            "height": vstream.get("height"),
        }

    async def _extract_frames(self, src: str, workdir: str) -> list[str]:
        outdir = os.path.join(workdir, "frames")
        os.makedirs(outdir, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            self._ff("ffmpeg"), "-y", "-loglevel", "error", "-i", src,
            "-vf", f"fps=1/{self.frame_interval_sec},scale='min(960,iw)':-2",
            "-frames:v", str(self.max_frames),
            os.path.join(outdir, "f_%03d.jpg"),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"抽帧失败: {err.decode(errors='ignore')[:300]}")
        return sorted(
            os.path.join(outdir, name)
            for name in os.listdir(outdir)
            if name.endswith(".jpg")
        )

    async def _extract_audio(self, src: str, workdir: str) -> str | None:
        if not self.stt_enabled:
            return None
        out = os.path.join(workdir, "audio16k.wav")
        proc = await asyncio.create_subprocess_exec(
            self._ff("ffmpeg"), "-y", "-loglevel", "error", "-i", src,
            "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", out,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1024:
            return out
        # 没有音轨或音频提取失败 → 返回 None 继续走视觉
        return None

    # ------------------------------------------------------------------ log
    def _log(self, message: str) -> None:
        try:
            from astrbot.api import logger

            logger.warning(f"[懂你意思] {message}")
        except Exception:  # noqa: BLE001
            pass
