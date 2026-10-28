# -*- coding: utf-8 -*-
"""本地 STT 客户端：管理一个长驻 worker 子进程（JSON-Lines IPC）。

为什么不用「同进程直调」：sherpa-onnx 解码崩溃/卡死会带崩 AstrBot 主进程，
而且 Qwen3-ASR 加载要 ~1GB 内存，独立子进程可隔离、可重启、可限制优先级。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

logger = logging.getLogger("astrbot")

CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


class LocalSTT:
    """子进程 worker 封装：串行队列 + 崩溃自动重启 + 超时。"""

    def __init__(
        self,
        python_exe: str,
        worker_script: str,
        model_dir: str,
        num_threads: int = 4,
        max_new_tokens: int = 512,
        hotwords: str = "",
        timeout_scale: float = 6.0,
    ) -> None:
        self.python_exe = python_exe
        self.worker_script = worker_script
        self.model_dir = model_dir
        self.num_threads = int(num_threads or 4)
        self.max_new_tokens = int(max_new_tokens or 512)
        self.hotwords = hotwords or ""
        self.timeout_scale = float(timeout_scale or 6.0)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._seq = 0

    # ---------------------------------------------------------------- worker
    def available(self) -> bool:
        return bool(
            self.python_exe
            and os.path.isfile(self.python_exe)
            and os.path.isfile(self.worker_script)
            and os.path.isdir(self.model_dir)
        )

    async def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if not self.available():
            raise RuntimeError("本地 STT 不可用（检查 python/worker/model 路径）")
        args = [
            self.python_exe,
            "-X",
            "utf8",
            "-u",
            self.worker_script,
            "--model-dir",
            self.model_dir,
            "--num-threads",
            str(self.num_threads),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.hotwords:
            args += ["--hotwords", self.hotwords]
        flags = CREATE_NO_WINDOW
        if sys.platform == "win32":
            flags |= BELOW_NORMAL_PRIORITY_CLASS
        t0 = time.time()
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=flags,
        )
        logger.info("[MemeSense] 本地 STT worker 已启动 pid=%s", getattr(self._proc, "pid", "?"))
        # 等 ping 回执（模型加载完成）
        resp = await self._roundtrip({"cmd": "ping"}, timeout=300)
        logger.info(
            "[MemeSense] 本地 STT 就绪：%.1fs %s",
            time.time() - t0,
            resp.get("info", ""),
        )

    async def _roundtrip(self, payload: dict, timeout: float) -> dict:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._seq += 1
        payload = dict(payload)
        payload.setdefault("id", str(self._seq))
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await asyncio.wait_for(self._proc.stdin.drain(), timeout=15)
        raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        if not raw:
            raise RuntimeError("本地 STT worker 无响应（可能已崩溃）")
        return json.loads(raw.decode("utf-8", "replace"))

    async def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
            self._proc = None

    # ---------------------------------------------------------------- public
    async def transcribe(self, audio_path: str, umo: str = "", timeout: float = 600) -> str:
        async with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    await self._ensure_worker()
                    resp = await self._roundtrip(
                        {"cmd": "transcribe", "path": audio_path}, timeout=timeout
                    )
                    if not resp.get("ok"):
                        raise RuntimeError(resp.get("error") or "transcribe failed")
                    return str(resp.get("text") or "")
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning("[MemeSense] 本地 STT 失败(第%s次): %s", attempt + 1, exc)
                    await self._kill()
            raise RuntimeError(str(last_exc))

    async def close(self) -> None:
        await self._kill()
