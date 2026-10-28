# -*- coding: utf-8 -*-
"""本地 STT worker（Qwen3-ASR-0.6B int8 + sherpa-onnx）。

由插件 stt_local.py 拉起，长驻运行，stdin/stdout 走 JSON Lines：
  → {"id":"1","cmd":"ping"}
  ← {"id":"1","ok":true,"info":"qwen3-asr 0.6B loaded 12.3s"}
  → {"id":"2","cmd":"transcribe","path":"C:\\tmp\\a.wav"}
  ← {"id":"2","ok":true,"text":"……","secs":8.4}

手动测试：
  D:\\DeepSeekQQ\\python\\python.exe -X utf8 -u stt_worker.py --model-dir <dir> --num-threads 4
  然后逐行输入 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def log(msg: str) -> None:
    sys.stderr.write("[stt_worker] " + msg + "\n")
    sys.stderr.flush()


def read_wav_16k(path: str, target_sr: int = 16000):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != target_sr and len(audio) > 0:
        # 简单线性插值重采样（音频量小，够用；不引 scipy）
        duration = len(audio) / float(sr)
        n_out = int(duration * target_sr)
        x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = target_sr
    return audio, sr


def build_recognizer(args):
    import sherpa_onnx

    model_dir = args.model_dir.rstrip("\\/")
    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=model_dir + r"\conv_frontend.onnx",
        encoder=model_dir + r"\encoder.int8.onnx",
        decoder=model_dir + r"\decoder.int8.onnx",
        tokenizer=model_dir + r"\tokenizer",
        num_threads=int(args.num_threads),
        provider="cpu",
        max_new_tokens=int(args.max_new_tokens),
        hotwords=args.hotwords or "",
    )
    log("loaded in %.1fs" % (time.time() - t0))
    return rec, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--hotwords", default="")
    args = ap.parse_args()

    rec = None
    load_secs = None

    def ensure():
        nonlocal rec, load_secs
        if rec is None:
            rec, load_secs = build_recognizer(args)
        return rec

    def emit(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            emit({"id": None, "ok": False, "error": "bad json"})
            continue
        rid = req.get("id")
        cmd = (req.get("cmd") or "").strip()
        try:
            if cmd == "ping":
                ensure()
                emit({"id": rid, "ok": True, "info": "qwen3-asr-0.6B loaded %.1fs" % (load_secs or 0)})
            elif cmd == "transcribe":
                recognizer = ensure()
                path = req.get("path") or ""
                t0 = time.time()
                audio, sr = read_wav_16k(path)
                secs = len(audio) / float(sr) if sr else 0.0
                stream = recognizer.create_stream()
                stream.accept_waveform(sr, audio)
                recognizer.decode_stream(stream)
                text = (stream.result.text or "").strip()
                emit(
                    {
                        "id": rid,
                        "ok": True,
                        "text": text,
                        "wav_seconds": round(secs, 2),
                        "secs": round(time.time() - t0, 2),
                        "rtf": round((time.time() - t0) / secs, 3) if secs else None,
                    }
                )
            elif cmd == "shutdown":
                emit({"id": rid, "ok": True})
                break
            else:
                emit({"id": rid, "ok": False, "error": "unknown cmd: " + cmd})
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc(file=sys.stderr)
            emit({"id": rid, "ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})
    log("worker exit")


if __name__ == "__main__":
    main()
