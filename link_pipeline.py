# -*- coding: utf-8 -*-
"""视频链接解析流水线：识别消息里的视频链接（B站/YouTube/抖音等），

1) 优先用平台 API 拿元信息（B站：标题/UP主/简介/标签/字幕）
2) 用 yt-dlp 下载限制清晰度/时长/体积的视频到临时目录
3) 之后交给 video_pipeline.VideoPipeline 做抽帧 + 语音转写
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from typing import Any

from astrbot.api import logger

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
URL_RE = re.compile(r"https?://[^\s<>\"'）)】\]，。；！？]+", re.I)
CREATE_NO_WINDOW = 0x08000000
DEFAULT_DOMAINS = [
    "bilibili.com", "b23.tv",
    "youtube.com", "youtu.be", "youtube-nocookie.com",
    "douyin.com", "v.douyin.com", "iesdouyin.com",
    "kuaishou.com", "v.kuaishou.com",
    "xiaohongshu.com", "xhslink.com",
    "weibo.com", "weibo.cn",
    "ixigua.com", "acfun.cn",
]


class LinkVideoError(Exception):
    pass


class LinkVideoPipeline:
    """消息链接 -> 本地视频文件 + 元信息 + 字幕。"""

    def __init__(self, config: dict | None = None, data_dir: str = ""):
        self.config = config or {}
        self.data_dir = data_dir or ""
        self.domains = [str(d).lower().strip() for d in
                        (self.config.get("allowed_domains") or DEFAULT_DOMAINS) if str(d).strip()]
        self.max_duration = float(self.config.get("max_duration_sec") or 900)
        self.max_size_mb = int(self.config.get("max_filesize_mb") or 200)
        self.quality = int(self.config.get("quality") or 480)
        self.timeout = int(self.config.get("timeout_sec") or 240)
        self.cookies_file = str(self.config.get("cookies_file") or "").strip()
        self.ytdlp_path = str(self.config.get("ytdlp_path") or "").strip()
        self.python_path = str(self.config.get("ytdlp_python") or "").strip()
        self._ytdlp: tuple[str, str] | None = None
        self._ytdlp_checked = False

    # ------------------------------------------------------------------ 链接识别
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def extract_urls(self, text: str) -> list[str]:
        if not text:
            return []
        found: list[str] = []
        for raw in URL_RE.findall(text):
            url = raw.rstrip(".,;:!?。，；：！？、…")
            try:
                host = (urllib.parse.urlparse(url).hostname or "").lower()
            except Exception:  # noqa: BLE001
                continue
            if not host:
                continue
            if any(host == d or host.endswith("." + d) or host.endswith(d) for d in self.domains):
                if url not in found:
                    found.append(url)
        return found[:3]

    # ------------------------------------------------------------------ yt-dlp 定位
    def _detect_ytdlp(self) -> tuple[str, str] | None:
        if self._ytdlp_checked:
            return self._ytdlp
        self._ytdlp_checked = True
        if self.ytdlp_path and os.path.isfile(self.ytdlp_path):
            self._ytdlp = ("exe", self.ytdlp_path)
            return self._ytdlp
        exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        if exe:
            self._ytdlp = ("exe", exe)
            return self._ytdlp
        for py in filter(None, [self.python_path,
                                os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python312", "python.exe"),
                                r"D:\DeepSeekQQ\python\python.exe"]):
            if not os.path.isfile(py):
                continue
            try:
                r = subprocess.run([py, "-m", "yt_dlp", "--version"], capture_output=True, text=True,
                                   creationflags=CREATE_NO_WINDOW, timeout=60)
                if r.returncode == 0:
                    self._ytdlp = ("py", py)
                    return self._ytdlp
            except Exception:  # noqa: BLE001
                continue
        logger.warning("[MemeSense] 未找到 yt-dlp（视频链接功能需要它）")
        return None

    def _ytdlp_cmd(self, extra: list[str]) -> list[str]:
        det = self._detect_ytdlp()
        if not det:
            raise LinkVideoError("没装 yt-dlp，无法打开视频链接")
        kind, path = det
        if kind == "py":
            base = [path, "-X", "utf8", "-m", "yt_dlp", "--no-warnings"]
        else:
            base = [path, "--no-warnings"]
        if self.cookies_file and os.path.isfile(self.cookies_file):
            base += ["--cookies", self.cookies_file]
        base += ["--socket-timeout", "20", "--retries", "2", "--no-playlist"]
        return base + extra

    # ------------------------------------------------------------------ 对外入口
    async def prepare(self, url: str, on_status: Any = None) -> dict:
        """解析链接 -> {meta, path, subtitle, workdir}；失败抛 LinkVideoError。"""
        workdir = tempfile.mkdtemp(prefix="meme_link_")
        try:
            if on_status:
                await on_status("我去看看这个视频，等我一下…")
            meta = await asyncio.to_thread(self._fetch_meta, url)
            duration = float(meta.get("duration") or 0)
            if duration > self.max_duration:
                raise LinkVideoError(f"这个视频有点长（{int(duration)} 秒），超过我一次能看的 {int(self.max_duration)} 秒上限")
            subtitle = ""
            if self.config.get("fetch_subtitle", True):
                subtitle = await asyncio.to_thread(self._fetch_subtitle, url, workdir)
            if on_status:
                await on_status("链接打开了，正在下载视频…")
            path = await asyncio.to_thread(self._download, url, workdir)
            return {"meta": meta, "path": path, "subtitle": subtitle, "workdir": workdir, "url": url}
        except Exception:
            self.cleanup(workdir)
            raise

    @staticmethod
    def cleanup(workdir: str) -> None:
        try:
            if workdir and os.path.isdir(workdir):
                shutil.rmtree(workdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 元信息
    def _http_json(self, url: str, headers: dict | None = None) -> dict | None:
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            logger.info("[MemeSense] 元信息请求失败 %s: %s", url[:90], exc)
            return None

    def _fetch_meta(self, url: str) -> dict:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if "bilibili" in host or "b23.tv" in host:
            meta = self._bilibili_meta(url)
            if meta:
                return meta
        return self._ytdlp_meta(url)

    # --- B站
    def _bilibili_resolve(self, url: str) -> tuple[str | None, str | None]:
        """返回 (bvid, page_url)；短链 b23.tv 解析。"""
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if "b23.tv" in host:
            url = self._follow(url)
        m = re.search(r"(BV[0-9A-Za-z]{10})", url)
        if m:
            return m.group(1), f"https://www.bilibili.com/video/{m.group(1)}"
        m = re.search(r"/video/av(\d+)", url)
        if m:
            data = self._http_json(f"https://api.bilibili.com/x/web-interface/view?aid={m.group(1)}",
                                   {"Referer": "https://www.bilibili.com"})
            if data and data.get("data", {}).get("bvid"):
                return data["data"]["bvid"], url
        return None, url

    def _follow(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.geturl()
        except Exception:  # noqa: BLE001
            return url

    def _bilibili_meta(self, url: str) -> dict | None:
        bvid, page = self._bilibili_resolve(url)
        if not bvid:
            return None
        headers = {"Referer": "https://www.bilibili.com"}
        view = self._http_json(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}", headers)
        if not view or view.get("code") != 0:
            return None
        d = view.get("data") or {}
        tags: list[str] = []
        tag_res = self._http_json(f"https://api.bilibili.com/x/tag/archive/tags?bvid={bvid}", headers)
        if tag_res and tag_res.get("code") == 0:
            tags = [t.get("tag_name") for t in (tag_res.get("data") or [])][:10]
        pages = d.get("pages") or [{}]
        meta = {
            "site": "bilibili",
            "id": bvid,
            "page_url": page,
            "title": d.get("title") or "",
            "author": ((d.get("owner") or {}).get("name")) or "",
            "description": (d.get("desc") or "")[:800],
            "duration": float(pages[0].get("duration") or d.get("duration") or 0),
            "tags": [t for t in tags if t],
            "stat": d.get("stat") or {},
            "cid": pages[0].get("cid"),
            "subtitle": "",
        }
        if meta["cid"]:
            pv = self._http_json(
                f"https://api.bilibili.com/x/player/v2?bvid={bvid}&cid={meta['cid']}", headers)
            if pv and pv.get("code") == 0:
                subs = (((pv.get("data") or {}).get("subtitle") or {}).get("subtitles") or [])
                for s in subs:
                    u = (s.get("subtitle_url") or "").strip()
                    if not u:
                        continue
                    if u.startswith("//"):
                        u = "https:" + u
                    body = self._http_json(u, {"Referer": page})
                    if body and body.get("body"):
                        text = " ".join(str(b.get("content") or "") for b in body["body"])
                        if text.strip():
                            meta["subtitle"] = text.strip()[:4000]
                            break
        return meta

    # --- yt-dlp 元信息
    def _ytdlp_meta(self, url: str) -> dict:
        cmd = self._ytdlp_cmd(["-J", "--skip-download", url])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=CREATE_NO_WINDOW, timeout=90)
        except Exception as exc:  # noqa: BLE001
            raise LinkVideoError(f"解析链接失败：{exc}") from exc
        if r.returncode != 0 or not (r.stdout or "").strip():
            raise LinkVideoError("这个链接我打不开（可能需要登录/已失效）")
        try:
            info = json.loads(r.stdout)
        except json.JSONDecodeError as exc:
            raise LinkVideoError("链接返回的内容不是视频") from exc
        return {
            "site": info.get("extractor_key") or info.get("extractor") or "",
            "id": info.get("id") or "",
            "page_url": info.get("webpage_url") or url,
            "title": info.get("title") or "",
            "author": info.get("uploader") or info.get("channel") or "",
            "description": (info.get("description") or "")[:800],
            "duration": float(info.get("duration") or 0),
            "tags": [str(t) for t in (info.get("tags") or [])][:10],
            "subtitle": "",
        }

    # ------------------------------------------------------------------ 字幕
    def _fetch_subtitle(self, url: str, workdir: str) -> str:
        patterns = [
            os.path.join(workdir, "*.vtt"),
            os.path.join(workdir, "*.srt"),
            os.path.join(workdir, "*.ass"),
        ]
        before = {p for pat in patterns for p in glob.glob(pat)}
        cmd = self._ytdlp_cmd(["--skip-download", "--write-subs", "--write-auto-subs",
                               "--sub-langs", "zh.*,zh-CN,zh-Hans,zh-Hant,en.*,ai-zh",
                               "--sub-format", "vtt/best",
                               "-o", os.path.join(workdir, "sub.%(ext)s"), url])
        try:
            subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=CREATE_NO_WINDOW, timeout=120)
        except Exception:  # noqa: BLE001
            return ""
        after = {p for pat in patterns for p in glob.glob(pat)}
        new = sorted(after - before)
        for path in new:
            try:
                text = _read_subtitle(path)
            except Exception:  # noqa: BLE001
                continue
            if text.strip():
                return text.strip()[:4000]
        return ""

    # ------------------------------------------------------------------ 下载
    def _download(self, url: str, workdir: str) -> str:
        outtmpl = os.path.join(workdir, "link.%(ext)s")
        fmt = (f"bv*[height<={self.quality}]+ba/b[height<={self.quality}]/"
               f"b[height<={self.quality}]/b")
        cmd = self._ytdlp_cmd([
            "-f", fmt, "--merge-output-format", "mp4",
            "--max-filesize", f"{self.max_size_mb}M",
            "--no-part", "-o", outtmpl, url,
        ])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               creationflags=CREATE_NO_WINDOW, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise LinkVideoError("下载超时了，换个短一点的视频试试") from exc
        files = [p for p in glob.glob(os.path.join(workdir, "link.*"))
                 if not p.endswith((".part", ".ytdl"))]
        if not files:
            tail = (r.stderr or r.stdout or "")[-400:].replace("\n", " ")
            raise LinkVideoError(f"下载失败：{tail[:200]}")
        files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return files[0]


def _read_subtitle(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\{[^}]*\}", "", line)
        if line:
            lines.append(line)
    seen: list[str] = []
    for line in lines:
        if not seen or seen[-1] != line:
            seen.append(line)
    return " ".join(seen)
