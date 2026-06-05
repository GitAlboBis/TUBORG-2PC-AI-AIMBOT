"""
USB Capture Card Module (Dual-PC Architecture).

Captures video from an HDMI capture card connected via USB.
The capture card receives video from an HDMI splitter connected to the gaming PC.

Key design:
  - Threaded capture to always have the latest frame (drop old frames)
  - Configurable pixel format (YUY2, MJPEG, NV12) to support different chipsets
  - Buffer size = 1 to prevent frame queue buildup
  - Same public API as DXGICapture for drop-in replacement
  - Chipset presets for popular capture cards

Supported chipsets / presets:
  - MS2130 (UGREEN, etc.)     — YUY2,  1080p60, USB 3.0, 4K passthrough
  - MS2109 (generic budget)    — MJPEG, 1080p30, USB 2.0
  - MS2131 (newer high-end)    — NV12,  4K30 or 1080p60, USB 3.0
  - Elgato HD60 S+             — NV12,  1080p60, USB 3.0
  - Generic / Auto             — auto-detect, tries YUY2 → MJPEG → raw
"""

import numpy as np
import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Dict

from exceptions import CaptureException

logger = logging.getLogger(__name__)


# ── Chipset Presets ────────────────────────────────────────────────
# Each preset defines: fourcc, resolution, max_fps, description, notes

CAPTURE_PRESETS: Dict[str, dict] = {
    "ms2130": {
        "fourcc": "YUY2",
        "width": 1920, "height": 1080,
        "fps": 60,
        "description": "MS2130 (UGREEN, etc.)",
        "notes": "USB 3.0 required. Native YUY2, 4K HDMI passthrough.",
    },
    "ms2109": {
        "fourcc": "MJPG",
        "width": 1920, "height": 1080,
        "fps": 30,
        "description": "MS2109 (generic budget)",
        "notes": "USB 2.0 compatible. MJPEG compression, max 1080p30.",
    },
    "ms2131": {
        "fourcc": "NV12",
        "width": 1920, "height": 1080,
        "fps": 60,
        "description": "MS2131 (high-end)",
        "notes": "USB 3.0 required. NV12 native, supports 4K30 capture.",
    },
    "elgato": {
        "fourcc": "NV12",
        "width": 1920, "height": 1080,
        "fps": 60,
        "description": "Elgato HD60 S+ / similar",
        "notes": "USB 3.0. NV12 output via DirectShow.",
    },
    "auto": {
        "fourcc": "auto",
        "width": 1920, "height": 1080,
        "fps": 60,
        "description": "Auto-detect (try YUY2 → MJPG → raw)",
        "notes": "Tries common formats in order. Use if unsure about chipset.",
    },
}

# FourCC format display names
FORMAT_LABELS = {
    "YUY2": "YUY2 (raw uncompressed)",
    "MJPG": "MJPEG (compressed)",
    "NV12": "NV12 (semi-planar YUV)",
    "auto": "Auto-detect",
}

# Supported resolutions for the GUI dropdown
CAPTURE_RESOLUTIONS = [
    (1920, 1080, "1080p"),
    (2560, 1440, "1440p"),
    (3840, 2160, "4K"),
    (1280, 720, "720p"),
    (1600, 900, "900p"),
]


class CaptureCardCapture:
    """
    USB capture card implementation using OpenCV VideoCapture.

    Captures HDMI video from gaming PC via an external capture card.
    Runs a background thread to continuously grab frames, ensuring
    the main loop always gets the most recent frame with minimal latency.

    Supports configurable pixel format and resolution to work with
    different chipsets (MS2130 YUY2, MS2109 MJPEG, MS2131/Elgato NV12).
    """

    def __init__(self, device_index: int = 0, fourcc: str = "YUY2",
                 width: int = 1920, height: int = 1080):
        """
        Args:
            device_index: USB video device index (usually 0 or 1).
                         On Windows, this maps to DirectShow device order.
            fourcc: Pixel format FourCC code ('YUY2', 'MJPG', 'NV12', 'auto').
            width:  Requested capture width.
            height: Requested capture height.
        """
        self.device_index = device_index
        self.requested_fourcc = fourcc.upper()
        self.requested_width = width
        self.requested_height = height
        self._cap = None
        self.initialized = False

        # Threaded capture state
        self._thread = None
        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_timestamp = 0.0
        # Drop-stale freshness ratchet for grab_latest (design.md §4.3).
        # The capture thread writes _frame_timestamp = time.perf_counter()
        # after every successful frame; grab_latest returns None when this
        # has not advanced past _consumed_timestamp since the last call.
        self._consumed_timestamp: float = -1.0

        # Actual resolution reported by card
        self._width = 0
        self._height = 0
        self._actual_fourcc = ""

        # Raw-YUY2 fast path (set during initialize): when True, the capture
        # thread stores the raw YUY2 buffer (no full-frame YUY2→BGR inside
        # cap.read()) and consumers convert only the pixels they actually use.
        self._raw_yuy2 = False

    def _as_yuy2(self, frame) -> Optional[np.ndarray]:
        """Normalize a raw-mode frame to a (H, W, 2) uint8 YUY2 view.

        Different OpenCV backends hand raw buffers back with different
        shapes — (H, W, 2), (H, 2W) or flat. Returns None if the buffer
        cannot be a YUY2 image of the negotiated resolution.
        """
        if frame is None or frame.dtype != np.uint8:
            return None
        h, w = self._height, self._width
        if frame.ndim == 3 and frame.shape == (h, w, 2):
            return frame
        if frame.size == h * w * 2:
            return frame.reshape(h, w, 2)
        return None

    def _try_fourcc(self, cap, fourcc_str: str) -> bool:
        """Attempt to set a specific FourCC on the capture device."""
        try:
            cc = cv2.VideoWriter_fourcc(*fourcc_str)
            cap.set(cv2.CAP_PROP_FOURCC, cc)
            # Verify it actually took
            actual = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_str = "".join([chr((actual >> (8 * i)) & 0xFF) for i in range(4)])
            if actual_str.strip('\x00').upper() == fourcc_str.upper():
                self._actual_fourcc = fourcc_str
                return True
        except Exception:
            pass
        return False

    def _auto_detect_format(self, cap) -> str:
        """Try common formats in order of preference for low latency."""
        for fmt in ["YUY2", "MJPG", "NV12"]:
            if self._try_fourcc(cap, fmt):
                logger.info(f"Auto-detected format: {fmt}")
                return fmt
        # If nothing matched, use whatever the device defaults to
        self._actual_fourcc = "raw"
        return "raw"

    def _open_device(self) -> bool:
        """
        Open the OpenCV ``VideoCapture`` backing this capture card.

        Tries the DirectShow backend first, then falls back to whatever
        backend OpenCV picks by default if DSHOW reports the device as
        closed. On a successful return ``self._cap`` holds the opened
        ``cv2.VideoCapture`` handle.

        Returns:
            ``True`` if the device was opened by either backend.
            ``False`` if the device could not be opened. This covers:
              - Both backends report ``isOpened() == False`` (total
                failure — ``initialize`` converts this into a
                ``CaptureException`` per R5.1).
              - The DSHOW driver raises while probing ``isOpened()``
                (driver misbehavior — the DSHOW handle is released and
                the fallback is intentionally skipped so we do not
                chain another potentially-misbehaving call).
        """
        # DSHOW backend first — best compatibility on Windows.
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)

        # Guard isOpened() — a misbehaving driver can raise here. Contain
        # the exception, release the native DSHOW handle so we do not
        # leak it, and report failure by returning False. The fallback
        # is intentionally skipped in this case.
        try:
            dshow_opened = self._cap.isOpened()
        except Exception as exc:
            logger.error(
                f"Capture card at device index {self.device_index}: "
                f"isOpened() raised on DSHOW: {exc}"
            )
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            return False

        if dshow_opened:
            return True

        # Fallback: try without explicit backend. Release the DSHOW
        # handle FIRST so we don't leak the native VideoCapture
        # resource when we reassign ``self._cap`` (Bug 4).
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = cv2.VideoCapture(self.device_index)

        return bool(self._cap.isOpened())

    def initialize(self, target_fps: int = 60,
                   region: Optional[Tuple[int, int, int, int]] = None,
                   silent: bool = False) -> bool:
        """
        Initialize the USB capture card.

        Args:
            target_fps: Target capture framerate (depends on capture card)
            region: Ignored for capture card (full frame always captured)
            silent: Disable console prints during initialization.

        Returns:
            True if initialization fully succeeds.
            False if a recoverable post-open failure occurred (e.g. the
            test read failed or the background thread could not start).

        Raises:
            CaptureException: If neither the DSHOW nor the fallback
                backend can open the capture device, or if the underlying
                driver raises while probing ``isOpened()`` (R5.1). The
                exception message identifies the ``device_index`` so
                callers can surface a descriptive error without having
                to re-derive it.
        """
        try:
            try:
                opened = self._open_device()
            except CaptureException:
                # Preserve any CaptureException from deeper helpers
                # verbatim — do not double-wrap.
                raise
            except Exception as exc:
                logger.error(
                    f"Cannot open capture card at device index "
                    f"{self.device_index}: {exc}"
                )
                raise CaptureException(
                    f"failed to open capture device {self.device_index}: {exc}"
                ) from exc

            if not opened:
                # Discriminate the two contained-failure modes produced
                # by ``_open_device``:
                #   - ``self._cap is None``: a misbehaving driver raised
                #     while probing ``isOpened()``; the handle was
                #     released inside the helper. Surface this as a
                #     ``False`` return so callers can retry/fall-back
                #     without catching exceptions (Bug 4 containment).
                #   - ``self._cap`` still set: both backends reported
                #     ``isOpened() == False``. This is the total-failure
                #     case — convert it into a structured
                #     ``CaptureException`` per R5.1.
                logger.error(
                    f"Cannot open capture card at device index {self.device_index}"
                )
                if self._cap is None:
                    return False
                raise CaptureException(
                    f"failed to open capture device {self.device_index}"
                )

            # Configure for low latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set pixel format
            if self.requested_fourcc == "AUTO":
                detected = self._auto_detect_format(self._cap)
                if not silent:
                    print(f"[CAPTURE] Auto-detected format: {detected}")
            else:
                if not self._try_fourcc(self._cap, self.requested_fourcc):
                    if not silent:
                        print(f"[CAPTURE] WARNING: {self.requested_fourcc} not accepted by device, using default")
                    self._actual_fourcc = "default"
                else:
                    if not silent:
                        print(f"[CAPTURE] Format set: {self.requested_fourcc}")

            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)

            # Read actual resolution
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Test read
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.error("Capture card: test read failed")
                self._cap.release()
                return False

            # Raw-YUY2 fast path: skip OpenCV's full-frame YUY2→BGR conversion
            # inside cap.read() (~1.6 ms/frame at 1080p on this ARM64 build,
            # measured) and convert only the consumed center crop in
            # grab_latest (~0.07 ms). DirectShow YUY2 is always top-down
            # (bottom-up DIBs only exist for RGB formats), so no flip handling
            # is needed. Probed defensively: if the backend hands back an
            # unusable raw layout, restore BGR conversion and carry on.
            self._raw_yuy2 = False
            if (self._actual_fourcc or "").upper() == "YUY2":
                try:
                    self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                    p_ret, probe = self._cap.read()
                    raw = self._as_yuy2(probe) if p_ret else None
                    if raw is not None:
                        bgr = cv2.cvtColor(
                            np.ascontiguousarray(raw), cv2.COLOR_YUV2BGR_YUY2
                        )
                        if bgr.shape == (self._height, self._width, 3):
                            self._raw_yuy2 = True
                except Exception:
                    self._raw_yuy2 = False
                if not self._raw_yuy2:
                    try:
                        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                        self._cap.read()  # flush one frame in restored mode
                    except Exception:
                        pass
                elif not silent:
                    print("[CAPTURE] Raw YUY2 mode: crop-only color conversion")

            # Start background capture thread
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="CaptureCard"
            )
            self._thread.start()

            self.initialized = True
            fmt_label = self._actual_fourcc or self.requested_fourcc
            logger.info(
                f"Capture card initialized: {self._width}x{self._height} "
                f"[{fmt_label}] @ device {self.device_index}"
            )
            if not silent:
                print(
                    f"[CAPTURE] OK: {self._width}x{self._height} "
                    f"[{fmt_label}] device={self.device_index} fps={target_fps}"
                )
            return True

        except CaptureException:
            # Propagate capture-layer failure signals to the caller.
            # The device could not be opened — there is nothing useful to
            # recover here, and silently swallowing this would convert a
            # structured failure back into a return-False surprise (R5.1).
            raise
        except Exception as e:
            logger.error(f"Failed to initialize capture card: {e}")
            return False

    def _capture_loop(self):
        """Background thread: continuously grab frames, keep only the latest."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_timestamp = time.perf_counter()
            else:
                time.sleep(0.001)

    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.

        Args:
            region: Optional (left, top, right, bottom) to crop

        Returns:
            BGR numpy array or None
        """
        if not self.initialized:
            return None

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            return None

        if self._raw_yuy2:
            raw = self._as_yuy2(frame)
            if raw is None:
                return None
            frame = cv2.cvtColor(
                np.ascontiguousarray(raw), cv2.COLOR_YUV2BGR_YUY2
            )

        if region:
            left, top, right, bottom = region
            return frame[top:bottom, left:right]

        return frame

    def grab_frame(self) -> Optional[np.ndarray]:
        """Return the full captured frame (BGR). Used by TargetTracker."""
        return self.grab()

    def grab_center_region(self, size: int = 320) -> Optional[np.ndarray]:
        """
        Grab a square region from the center of the captured frame.

        In dual-PC mode, the "screen center" is the center of the
        capture card frame (which mirrors the gaming PC screen).

        Args:
            size: Side length of square capture region in pixels

        Returns:
            BGR numpy array of center region, or None
        """
        frame = self.grab()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        half = size // 2
        cx, cy = w // 2, h // 2

        top = max(0, cy - half)
        bottom = min(h, cy + half)
        left = max(0, cx - half)
        right = min(w, cx + half)

        return frame[top:bottom, left:right]

    def grab_latest(self, size: int = 320) -> Optional[np.ndarray]:
        """Drop-stale variant of grab_center_region (design.md §4.3, R2.3).

        Returns the most recent center-region crop only if the capture
        thread has produced a new frame since the previous call to
        grab_latest. Otherwise returns None. The capture initialization
        path and _capture_loop semantics are unchanged — this is a pure
        consumer-side freshness gate.
        """
        if not self.initialized:
            return None
        with self._frame_lock:
            ts = self._frame_timestamp
            if ts <= self._consumed_timestamp:
                return None
            frame = self._latest_frame
            self._consumed_timestamp = ts
        if frame is None:
            return None

        if self._raw_yuy2:
            # Crop first, convert only the consumed pixels (YUY2→BGR on
            # 416² costs ~0.07 ms vs ~1.6 ms full-frame, measured).
            raw = self._as_yuy2(frame)
            if raw is None:
                return None
            h, w = raw.shape[:2]
            half = size // 2
            cx, cy = w // 2, h // 2
            top = max(0, cy - half)
            bottom = min(h, cy + half)
            left = max(0, cx - half) & ~1  # YUY2 macropixel = 2 px: even col
            right = min(w, cx + half)
            if (right - left) % 2:
                right -= 1
            crop = np.ascontiguousarray(raw[top:bottom, left:right])
            try:
                return cv2.cvtColor(crop, cv2.COLOR_YUV2BGR_YUY2)
            except cv2.error:
                return None

        h, w = frame.shape[:2]
        half = size // 2
        cx, cy = w // 2, h // 2
        top = max(0, cy - half)
        bottom = min(h, cy + half)
        left = max(0, cx - half)
        right = min(w, cx + half)
        return frame[top:bottom, left:right]

    def grab_crosshair_region(self, size: int = 2) -> Optional[np.ndarray]:
        """
        Grab a tiny region at exact center for HSV triggerbot.

        Args:
            size: Half-size of capture region (total = 2*size x 2*size pixels)

        Returns:
            BGR numpy array of crosshair pixels, or None
        """
        frame = self.grab()
        if frame is None:
            return None

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        top = max(0, cy - size)
        bottom = min(h, cy + size)
        left = max(0, cx - size)
        right = min(w, cx + size)

        return frame[top:bottom, left:right]

    def get_resolution(self) -> Tuple[int, int]:
        """Get the capture card resolution."""
        return (self._width, self._height)

    def get_format_info(self) -> dict:
        """Get capture card format information."""
        return {
            "fourcc": self._actual_fourcc or self.requested_fourcc,
            "resolution": f"{self._width}x{self._height}",
            "device_index": self.device_index,
            "initialized": self.initialized,
        }

    def stop(self):
        """Stop the capture stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def cleanup(self):
        """Release all capture resources."""
        self.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self._latest_frame = None
        self.initialized = False
        logger.info("Capture card cleaned up")


__all__ = ['CaptureCardCapture', 'CAPTURE_PRESETS', 'CAPTURE_RESOLUTIONS', 'FORMAT_LABELS']
