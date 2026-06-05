"""Real-hardware latency benchmark for the 2PC pipeline (Surface Pro 11 / X Elite).

Measures, on the actual machine:
  1. AI inference (process_frame) for each available model via the production
     AIVisionEngine path (QNN EP -> Hexagon NPU), incl. session load time.
  2. Preprocess-only cost (resize+cvtColor+fp16 normalize).
  3. YUY2->BGR 1080p conversion cost (what cv2.read() does in the capture thread).
  4. XXTEA encrypt + packet build + UDP sendto (loopback) — the move() hot path.
  5. LAN RTT to the KmBox (ICMP).
  6. (If capture card present) fresh-frame rate + combined capture+inference loop.

Usage:  python tools/bench_latency.py [--skip-capture] [--models a,b,c]
"""
import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(2)  # warnings: shows CPU-fallback notices
except Exception:
    pass

import config as cfg  # noqa: E402

pc = time.perf_counter


def stats(ms_list):
    s = sorted(ms_list)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "p50": s[n // 2],
        "p90": s[int(n * 0.90)],
        "p99": s[min(n - 1, int(n * 0.99))],
        "max": s[-1],
    }


def fmt(label, st):
    return (f"  {label:<28} mean={st['mean']:7.2f}  p50={st['p50']:7.2f}  "
            f"p90={st['p90']:7.2f}  p99={st['p99']:7.2f}  max={st['max']:7.2f} ms  (n={st['n']})")


def bench_engine(model_path: str, n: int = 200, warmup: int = 30):
    from engines.ai_engine import AIVisionEngine

    config = cfg.load_config()
    ai_cfg = dict(config.get("ai_engine", {}))
    ai_cfg["model_path"] = model_path

    eng = AIVisionEngine(ai_cfg, shared_state=None)
    t0 = pc()
    try:
        ok = eng.load_model()
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return
    load_s = pc() - t0
    if not ok:
        print("  LOAD returned False")
        return

    backend = eng._backend
    prov = imgsz = in_shape = None
    provider_obj = eng._qnn_provider or eng._dml_provider
    if provider_obj is not None:
        prov = getattr(provider_obj, "provider_used", None)
        imgsz = getattr(provider_obj, "imgsz", None)
        sess = getattr(provider_obj, "session", None)
        if sess is not None:
            try:
                in_shape = sess.get_inputs()[0].shape
            except Exception:
                pass
    print(f"  load={load_s:.2f}s  backend={backend}  provider={prov}  "
          f"imgsz={imgsz}  session_input_shape={in_shape}")

    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 255, (416, 416, 3), dtype=np.uint8) for _ in range(8)]

    for i in range(warmup):
        eng.process_frame(frames[i % 8])

    ts = []
    for i in range(n):
        t = pc()
        eng.process_frame(frames[i % 8])
        ts.append((pc() - t) * 1000.0)
    print(fmt("process_frame (e2e)", stats(ts)))

    if provider_obj is not None and hasattr(provider_obj, "preprocess"):
        pre = []
        for i in range(150):
            t = pc()
            provider_obj.preprocess(frames[i % 8])
            pre.append((pc() - t) * 1000.0)
        print(fmt("preprocess only", stats(pre)))

    try:
        eng.release()
    except Exception:
        pass


def bench_cvtcolor():
    rng = np.random.default_rng(0)
    yuy2 = rng.integers(0, 255, (1080, 1920, 2), dtype=np.uint8)
    # warm
    for _ in range(10):
        cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
    ts = []
    for _ in range(100):
        t = pc()
        cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
        ts.append((pc() - t) * 1000.0)
    print(fmt("cvtColor YUY2->BGR 1080p", stats(ts)))
    # crop-only variant: convert just the central 416x416 (+2px alignment slack)
    crop = yuy2[332:748, 752:1168].copy()
    for _ in range(10):
        cv2.cvtColor(crop, cv2.COLOR_YUV2BGR_YUY2)
    ts = []
    for _ in range(100):
        t = pc()
        cv2.cvtColor(crop, cv2.COLOR_YUV2BGR_YUY2)
        ts.append((pc() - t) * 1000.0)
    print(fmt("cvtColor YUY2->BGR 416crop", stats(ts)))


def bench_xxtea_udp():
    import socket
    from input.kmbox_net_driver import _Encryptor, _pack_header, _pack_mouse

    enc = _Encryptor(0xB6860C3D)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", 49999)  # nobody listening; sendto still measures syscall
    import random as _random

    ts = []
    for i in range(2000):
        t = pc()
        header = _pack_header(0xB6860C3D, _random.getrandbits(32), i, 0xAEDE7345)
        payload = header + _pack_mouse(0, 3, -2, 0)
        payload = enc.encrypt(payload)
        sock.sendto(payload, addr)
        ts.append((pc() - t) * 1000.0)
    print(fmt("move() build+encrypt+sendto", stats(ts)))
    sock.close()


def ping_kmbox(ip: str):
    try:
        out = subprocess.run(
            ["ping", "-n", "5", "-w", "300", ip],
            capture_output=True, text=True, timeout=15,
        )
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        for l in lines[-3:]:
            print("  " + l)
    except Exception as e:
        print(f"  ping failed: {e}")


def bench_capture(combined_model: str | None):
    from capture import CaptureCardCapture
    from capture.capture_card import CAPTURE_PRESETS

    config = cfg.load_config()
    cap_cfg = config.get("capture", {})
    preset = CAPTURE_PRESETS.get(cap_cfg.get("chipset_preset", "ms2130"),
                                 CAPTURE_PRESETS["auto"])
    capture = CaptureCardCapture(
        device_index=cap_cfg.get("device_index", 0),
        fourcc=cap_cfg.get("fourcc", preset["fourcc"]),
        width=cap_cfg.get("resolution_width", preset.get("width", 1920)),
        height=cap_cfg.get("resolution_height", preset.get("height", 1080)),
    )
    try:
        ok = capture.initialize(target_fps=int(cap_cfg.get("fps_cap", 60)), silent=False)
    except Exception as e:
        print(f"  capture init failed ({e}) — skipping capture benchmarks")
        return
    if not ok:
        print("  capture init returned False — skipping")
        return

    # Fresh-frame cadence + consumer pickup delay
    intervals, pickup = [], []
    last_fresh = None
    t_end = pc() + 4.0
    while pc() < t_end:
        f = capture.grab_latest(size=416)
        now = pc()
        if f is None:
            time.sleep(0.0005)
            continue
        pickup.append((now - capture._frame_timestamp) * 1000.0)
        if last_fresh is not None:
            intervals.append((now - last_fresh) * 1000.0)
        last_fresh = now
    if intervals:
        print(fmt("fresh-frame interval", stats(intervals)))
        print(fmt("grab pickup delay", stats(pickup)))
        print(f"  effective capture fps ~ {1000.0 / (sum(intervals)/len(intervals)):.1f}")
    else:
        print("  no frames received in 4s")

    # Combined loop: replicate main_simple hot loop (capture + inference)
    if combined_model:
        from engines.ai_engine import AIVisionEngine
        ai_cfg = dict(config.get("ai_engine", {}))
        ai_cfg["model_path"] = combined_model
        eng = AIVisionEngine(ai_cfg, shared_state=None)
        try:
            if eng.load_model():
                loop_ts, infer_ts = [], []
                t_last = None
                t_end = pc() + 8.0
                while pc() < t_end:
                    f = capture.grab_latest(size=416)
                    if f is None:
                        time.sleep(0.001)
                        continue
                    t0 = pc()
                    eng.process_frame(f)
                    infer_ts.append((pc() - t0) * 1000.0)
                    if t_last is not None:
                        loop_ts.append((pc() - t_last) * 1000.0)
                    t_last = pc()
                if loop_ts:
                    print(fmt("COMBINED loop interval", stats(loop_ts)))
                    print(fmt("COMBINED inference", stats(infer_ts)))
                    print(f"  effective end-to-end loop fps ~ {1000.0 / (sum(loop_ts)/len(loop_ts)):.1f}")
        finally:
            try:
                eng.release()
            except Exception:
                pass

    capture.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("BENCH 1: model metadata + inference latency (production engine path)")
    print("=" * 78)
    models = [
        "./models/yolov8m-valorant-detection.onnx",   # current config
        "./models/v11n-416-2.onnx",
        "./models/v11n-416-2-fp16.onnx",
    ]
    if not args.skip_models:
        for m in models:
            print(f"\nMODEL: {m}")
            bench_engine(m)

    print()
    print("=" * 78)
    print("BENCH 2: capture-thread YUY2->BGR conversion cost (CPU, ARM64)")
    print("=" * 78)
    bench_cvtcolor()

    print()
    print("=" * 78)
    print("BENCH 3: KmBox move() packet path (XXTEA + UDP sendto)")
    print("=" * 78)
    bench_xxtea_udp()

    print()
    print("=" * 78)
    print("BENCH 4: LAN RTT to KmBox")
    print("=" * 78)
    config = cfg.load_config()
    ping_kmbox(config.get("input", {}).get("kmbox_net", {}).get("ip", "192.168.2.188"))

    print()
    print("=" * 78)
    print("BENCH 5: capture card (if connected) + combined loop")
    print("=" * 78)
    if args.skip_capture:
        print("  skipped (--skip-capture)")
    else:
        bench_capture(combined_model="./models/yolov8m-valorant-detection.onnx")

    print("\nDone.")


if __name__ == "__main__":
    main()
