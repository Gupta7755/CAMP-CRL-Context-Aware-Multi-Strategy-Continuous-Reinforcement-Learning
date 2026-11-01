"""Live camera previews for the dashboard's Cameras page, independent of any simulation run.

Each started preview runs one CameraPipeline (detector + tracker + health, no scripted faults) in its
own thread on the wall clock. It reads frames as fast as the camera sends them (up to `max_fps`) and
runs detection at most `detect_hz` times per second on those same frames. Viewers get the latest raw
frame, optionally with the latest detections drawn on it, as a live MJPEG stream (`frames()`) or a
single JPEG (`jpeg()`); encoding happens per viewer, so an unwatched preview costs no JPEG work.
A preview nobody has fetched for `idle_stop_s` stops itself, so a closed browser tab does not keep
the phone streaming and the CPU busy.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from camp_crl.vision.config import VisionConfig, build_pipelines
from camp_crl.vision.pipeline import CameraPipeline
from camp_crl.vision.types import ApproachPerception

log = logging.getLogger(__name__)


class _Preview:
    def __init__(self, approach: str, pipeline: CameraPipeline) -> None:
        self.approach, self.pipeline = approach, pipeline
        self.stop = threading.Event()
        self.jpeg: bytes | None = None
        self.perception: dict | None = None
        self.error: str | None = None
        self.started = time.time()
        self.last_fetch = time.monotonic()
        self.frames = 0
        self.thread: threading.Thread | None = None


class CameraPreviews:
    def __init__(self, vision_provider: Callable[[], VisionConfig], max_fps: float = 30.0, detect_hz: float = 12.0,
                 max_width: int = 1280, idle_stop_s: float = 30.0) -> None:
        self.vision_provider = vision_provider
        self.max_fps, self.detect_hz, self.max_width, self.idle_stop_s = max_fps, detect_hz, max_width, idle_stop_s
        self._lock = threading.Lock()
        self._previews: dict[str, _Preview] = {}

    def start(self, approach: str) -> dict:
        """(Re)start the preview of one approach with its currently configured source."""
        self.stop(approach)
        vcfg = self.vision_provider()
        if approach not in vcfg.cameras:
            raise ValueError(f"no camera configured for approach {approach}")
        one = vcfg.model_copy(update={"cameras": {approach: vcfg.cameras[approach]}})
        pipeline = build_pipelines(one, seed=0, with_faults=False)[approach]
        pipeline.retry_max_s = min(pipeline.retry_max_s, 5.0)  # someone is watching: reconnect quickly
        pipeline.detect_interval_s = 1.0 / self.detect_hz
        pv = _Preview(approach, pipeline)
        pv.thread = threading.Thread(target=self._loop, args=(pv,), name=f"preview-{approach}", daemon=True)
        with self._lock:
            self._previews[approach] = pv
        pv.thread.start()
        return self.status()[approach]

    def _loop(self, pv: _Preview) -> None:
        period = 1.0 / self.max_fps
        try:
            while not pv.stop.is_set():
                t0 = time.monotonic()
                if t0 - pv.last_fetch > self.idle_stop_s:
                    log.info("preview %s stopped: not viewed for %.0f s", pv.approach, self.idle_stop_s)
                    break
                per = pv.pipeline.step(t0)  # blocks until the stream delivers a frame (or times out)
                with self._lock:
                    pv.perception = {**per.to_dict(), "detect_ms": None if pv.pipeline.detect_ms is None
                                     else round(pv.pipeline.detect_ms, 1),
                                     "stream_fps": round(pv.pipeline.raw_fps, 1)}
                    pv.frames = pv.pipeline.raw_frames
                pv.stop.wait(max(0.0, period - (time.monotonic() - t0)))
        except Exception as exc:  # report on the page instead of dying silently
            log.exception("preview %s crashed", pv.approach)
            with self._lock:
                pv.error = str(exc)
        finally:
            pv.pipeline.close()
            pv.stop.set()

    def stop(self, approach: str) -> None:
        with self._lock:
            pv = self._previews.pop(approach, None)
        if pv is not None:
            pv.stop.set()
            if pv.thread is not None and pv.thread is not threading.current_thread():
                pv.thread.join(timeout=8.0)

    def stop_all(self) -> None:
        for a in list(self._previews):
            self.stop(a)

    def _encode(self, approach: str, annotate: bool, after: int = 0) -> tuple[int, bytes] | None:
        """Latest frame newer than frame number `after`, as JPEG; marks the preview as watched."""
        import cv2

        with self._lock:
            pv = self._previews.get(approach)
            if pv is None:
                return None
            pv.last_fetch = time.monotonic()
        live = pv.pipeline.live_image(annotate)
        if live is None or live[0] <= after:
            return None
        n, img = live
        h, w = img.shape[:2]
        if w > self.max_width:
            img = cv2.resize(img, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return (n, buf.tobytes()) if ok else None

    def jpeg(self, approach: str, annotate: bool = True) -> bytes | None:
        got = self._encode(approach, annotate)
        return got[1] if got else None

    def latest(self, approach: str) -> tuple[ApproachPerception | None, float]:
        """(perception, longest measured stop in s) of a running preview, and mark it as in use."""
        with self._lock:
            pv = self._previews.get(approach)
            if pv is None or pv.stop.is_set():
                return None, 0.0
            pv.last_fetch = time.monotonic()
        return pv.pipeline.perception(time.monotonic()), pv.pipeline.max_wait_s

    def running(self, approach: str) -> bool:
        with self._lock:
            pv = self._previews.get(approach)
            return pv is not None and not pv.stop.is_set()

    def frames(self, approach: str, annotate: bool = True, max_frames: int | None = None):
        """MJPEG body generator: yields each new frame once, until the preview stops (or max_frames)."""
        last, sent, idle_since = 0, 0, time.monotonic()
        while self.running(approach) and (max_frames is None or sent < max_frames):
            got = self._encode(approach, annotate, after=last)
            if got is None:
                if time.monotonic() - idle_since > 10:  # keep the connection alive while a camera reconnects
                    with self._lock:
                        pv = self._previews.get(approach)
                        if pv is not None:
                            pv.last_fetch = time.monotonic()
                    idle_since = time.monotonic()
                time.sleep(0.01)
                continue
            last, jpg = got
            idle_since = time.monotonic()
            sent += 1
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")

    def status(self) -> dict[str, dict]:
        with self._lock:
            return {a: {"running": not pv.stop.is_set(), "frames": pv.frames, "error": pv.error,
                        "source": pv.pipeline.source.name, "perception": pv.perception,
                        "events": [{"t": round(t - pv.pipeline.events[0][0], 1), "event": m}
                                   for t, m in pv.pipeline.events[-5:]] if pv.pipeline.events else []}
                    for a, pv in self._previews.items()}


def probe_source(source: str, timeout_ms: int = 5000) -> dict:
    """Open a source and read one frame: {ok, width, height, ms, error}. Nothing is kept running."""
    from camp_crl.vision.sources import OpenCVSource, SourceError, SyntheticCamera

    t0 = time.perf_counter()
    src = SyntheticCamera(400.0) if source == "synthetic" else OpenCVSource(source, timeout_ms=timeout_ms)
    try:
        src.open(0.0)
        frame = None
        for _ in range(3):  # a stream may need a moment before the first decodable frame
            frame = src.read(0.1)
            if frame is not None:
                break
        if frame is None:
            raise SourceError("connected but no frame received")
        h, w = frame.image.shape[:2]
        return {"ok": True, "width": w, "height": h, "ms": round((time.perf_counter() - t0) * 1000), "error": None}
    except Exception as exc:
        return {"ok": False, "width": None, "height": None, "ms": round((time.perf_counter() - t0) * 1000),
                "error": str(exc)}
    finally:
        src.close()
