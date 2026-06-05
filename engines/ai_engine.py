"""
AI Vision Engine (Module 2) — YOLOv11 inference pipeline.

Backend selection (automatic):
    1. QNN (Hexagon NPU on Snapdragon X / ARM64) via onnxruntime-qnn — < 5ms target
    2. DirectML (AMD GPU) via onnxruntime-directml  — 2.3ms @ RX 7800 XT
    3. TensorRT (NVIDIA GPU) via Ultralytics         — <2ms @ RTX 3060+
    4. ONNX CPU via Ultralytics                      — 8ms fallback
    5. PyTorch CPU via Ultralytics                    — 17ms last resort

For ``.onnx`` model paths, selection is delegated to the pure
``engines.ep_selector.select_provider`` cascade (Reqs 3.x). The legacy
``.pt`` / ``.engine`` short-circuit goes straight through ``_try_ultralytics``.

Class mapping (Leaf48 ver-2):
    0 = enemy
    1 = ally

PRD Reference: MODULE 2 lines 113-180
"""

import numpy as np
import cv2
import os
import platform
import time
import logging
from collections import deque
from typing import List, Optional
from dataclasses import dataclass

from exceptions import AIEngineException, ValidationException
from utils.validation import validate_frame

logger = logging.getLogger(__name__)


# Once-per-process latch for the "both QNN and DirectML EPs present" WARN
# (Req 7.6). Module-level so it survives multiple ``load_model`` calls within
# the same Python process.
_qnn_dml_coexistence_warned: bool = False

# Once-per-process latch for the arm64 ``pip install onnxruntime-qnn`` install
# hint (Req 7.3). Module-level so the hint fires exactly once across multiple
# ``load_model`` invocations within the same Python process.
_qnn_hint_emitted: bool = False

# Valid backend identifiers per Req 5.4a strict-equality contract. The
# AIVisionEngine ``ai_backend`` shared-state key — and ``self._backend`` —
# are constrained to this set at all times.
_VALID_BACKENDS = frozenset({"qnn", "directml", "ultralytics", "none"})


def _normalize_host_arch() -> str:
    """Return ``"arm64"``, ``"x86_64"``, or ``"other"`` for the host machine."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("amd64", "x86_64"):
        return "x86_64"
    return "other"


def _snapshot_available_providers() -> List[str]:
    """Lazily query ``onnxruntime.get_available_providers``; tolerate ``ImportError``.

    Used by ``load_model`` to feed ``select_provider`` and to detect the
    QNN/DirectML coexistence condition (Req 7.6).

    Triggers the once-per-process QNN plugin-EP registration via
    :func:`engines.qnn_provider._ensure_qnn_registered` so the canonical
    ``"QNNExecutionProvider"`` identifier is present in the snapshot whenever
    ``onnxruntime-qnn`` is installed alongside a plugin-EP-aware ORT build.
    Without this hook the cascade picker would never see QNN as available
    on Snapdragon X Elite hosts (ORT 1.20+ does not auto-register
    out-of-tree EPs at import time).
    """
    try:
        import onnxruntime as ort  # local import — Req 7.5 hygiene
    except ImportError:
        return []
    # Best-effort QNN plugin registration. Failures are silent — they fold
    # cleanly into the "QNN not available" branch of the cascade.
    try:
        from engines.qnn_provider import _ensure_qnn_registered
        _ensure_qnn_registered()
    except Exception:  # noqa: BLE001
        pass
    try:
        return list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 — defensive
        return []


@dataclass
class Detection:
    """Single detection result from YOLO inference."""
    class_id: int        # 0=enemy, 1=ally
    class_name: str      # "enemy" or "ally"
    x: float             # bbox center X (in capture coords)
    y: float             # bbox center Y
    w: float             # bbox width
    h: float             # bbox height
    confidence: float    # 0.0-1.0


# Model search order — FP32 ONNX > TRT > PT.
# Note: a post-conversion FP16 ONNX (v11n-416-2-fp16.onnx) was tested but
# proved slower on QNN HTP than the FP32 baseline because the
# onnxconverter-common toolchain leaves Resize nodes in FP32 and inserts
# Cast nodes around them, increasing intermediate-tensor DMA bandwidth.
# A real FP16 win requires a native PyTorch export with ``half=True``
# (ultralytics ``model.export(format='onnx', half=True)``), which needs the
# original .pt artifact we don't currently have. Until then, prefer FP32.
MODEL_SEARCH_PATHS = [
    "./models/yolov8m-valorant-detection.onnx",
    "./models/v11n-416-2.onnx",
    "./models/v11n.onnx",
    "./models/v11n-416-2-fp16.onnx",
    "./models/v11n-fp16.onnx",
    "./models/valorant_v11n.onnx",
    "./models/v11n-416-2.engine",
    "./models/v11n.engine",
    "./models/v11n-416-2.pt",
    "./models/v11n.pt",
]

# NOTE: this mapping matches yolov8m-valorant-detection.onnx. The v11n-416-2
# models use {0: 'enemy', 1: 'ally'} (see ONNX metadata 'names'). class_name
# is cosmetic — targeting uses class_id via config target_classes, which MUST
# be realigned when switching models (v11n: [0], yolov8m: [1]).
CLASS_NAMES = {
    0: "dropped_spike",
    1: "enemy",
    2: "planted_spike",
    3: "teammate",
}


class AIVisionEngine:
    """
    AI Vision Engine with automatic backend selection.

    Pipeline:
        1. Capture center region (416px)
        2. YOLO11n inference (DirectML/TRT/ONNX/PT)
        3. NMS + confidence filter
        4. Target selection (closest to crosshair + headshot bias)

    Benchmarked: 2.3ms avg on AMD RX 7800 XT (DirectML)
    """

    def __init__(self, config: dict, shared_state=None):
        """Initialize AI engine with optional shared state integration.
        
        Args:
            config: Configuration dictionary with AI engine settings
            shared_state: Optional SharedState instance for live updates
        """
        # Stash a reference to the original ai_engine config dict so
        # ``load_model`` can read keys (e.g. ``execution_provider``) without a
        # second plumb-through. This is read-only from this class's POV.
        self.config = config

        self.enabled = config.get('enabled', True)
        self.model_path = config.get('model_path', '')
        self.confidence_threshold = config.get('confidence', 0.55)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.capture_size = config.get('capture_size', 416)
        self.headshot_bias = config.get('headshot_bias', 0.30)
        self.target_classes = config.get('target_classes', [0])

        # Backend state
        self._backend = None       # 'qnn' | 'directml' | 'ultralytics' | None
        self._qnn_provider = None  # QNNProvider instance (set by 6.2 wiring)
        self._dml_provider = None  # DirectMLProvider instance
        self._ul_model = None      # Ultralytics YOLO instance
        self._ul_device = 'cpu'
        self.model_loaded = False
        self._inference_times: deque = deque(maxlen=100)
        self.last_raw_detections = []
        
        # Shared state integration
        self._shared_state = shared_state
        self._update_state_metrics()

    def _find_model(self) -> str:
        """Find best available model file, looking both at working directory and package root."""
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Check user configured model_path
        if self.model_path:
            if os.path.exists(self.model_path):
                return self.model_path
            resolved = os.path.normpath(os.path.join(package_root, self.model_path))
            if os.path.exists(resolved):
                return resolved

        # 2. Check MODEL_SEARCH_PATHS
        for path in MODEL_SEARCH_PATHS:
            if os.path.exists(path):
                return path
            resolved = os.path.normpath(os.path.join(package_root, path))
            if os.path.exists(resolved):
                return resolved
        return ""

    def load_model(self) -> bool:
        """
        Load model with automatic backend selection.

        For ``.pt`` / ``.engine`` paths: short-circuit to the existing
        Ultralytics path (preserves Req 5.7 untouched code).

        For ``.onnx`` paths: delegate to ``engines.ep_selector.select_provider``,
        which evaluates a host-arch + override-aware cascade. Selection mapping:

            cascade key        public backend_name (gap-resolution #11)
            -------------      ---------------------------------------
            "qnn"              "qnn"
            "directml"         "directml"
            "cpu"              "ultralytics"

        On selector exhaustion (every candidate failed), ``AIEngineException``
        is raised after writing ``shared_state['ai_backend'] = 'none'``
        (Req 4.7, 5.4a).
        """
        model_path = self._find_model()
        if not model_path:
            error_msg = (
                "No model found. Run: python scripts/export_tensorrt.py\n"
                f"  Searched: {MODEL_SEARCH_PATHS}"
            )
            logger.error(error_msg)
            raise AIEngineException(error_msg)

        ext = os.path.splitext(model_path)[1].lower()

        # .pt / .engine short-circuit — bypass select_provider entirely so the
        # existing Ultralytics behaviour is preserved byte-for-byte.
        if ext != '.onnx':
            try:
                return self._try_ultralytics(model_path, ext)
            except AIEngineException:
                self._set_backend(None)
                raise

        # ----- ONNX path: cascade via select_provider -----
        from engines.ep_selector import select_provider

        host_arch = _normalize_host_arch()
        available = _snapshot_available_providers()

        # Req 7.3: on arm64 hosts where QNNExecutionProvider is unavailable,
        # log the install hint exactly once per process. Uses the module-level
        # ``_qnn_hint_emitted`` latch so repeated ``load_model`` calls do not
        # spam the log. ``has_qnn()`` is consulted (rather than the local
        # ``available`` snapshot) to keep the gating logic in one place — the
        # qnn_provider helper handles ``ImportError`` internally.
        global _qnn_hint_emitted
        if host_arch == "arm64" and not _qnn_hint_emitted:
            try:
                from engines.qnn_provider import has_qnn
                if not has_qnn():
                    logger.info("pip install onnxruntime-qnn")
                    _qnn_hint_emitted = True
            except ImportError:
                # qnn_provider module itself is unavailable — still emit the
                # hint so the user can resolve the missing dependency.
                logger.info("pip install onnxruntime-qnn")
                _qnn_hint_emitted = True

        # Req 7.6: warn-once if both EPs are present in the same env.
        global _qnn_dml_coexistence_warned
        if (
            "QNNExecutionProvider" in available
            and "DmlExecutionProvider" in available
            and not _qnn_dml_coexistence_warned
        ):
            logger.warning(
                "Both QNN and DirectML EPs present — proceeding with cascade rules"
            )
            _qnn_dml_coexistence_warned = True

        config_override = self.config.get("execution_provider", "auto")

        candidate_factories = {
            "qnn": lambda: self._build_qnn_provider(model_path),
            "directml": lambda: self._build_dml_provider(model_path),
            "cpu": lambda: self._build_ultralytics_provider(model_path, ext),
        }

        try:
            provider, cascade_key = select_provider(
                host_arch=host_arch,
                available_providers=available,
                config_override=config_override,
                candidate_factories=candidate_factories,
            )
        except AIEngineException:
            # Cascade exhausted — clear backend (writes ai_backend='none' to
            # shared_state per Req 5.4a) and re-raise unchanged (Req 4.7).
            self._set_backend(None)
            raise

        # Map cascade key "cpu" to public backend_name "ultralytics"
        # (gap-resolution #11).
        backend_name = "ultralytics" if cascade_key == "cpu" else cascade_key

        self._attach_provider(provider, backend_name)
        self._set_backend(backend_name)
        self.model_loaded = True
        return True

    # ------------------------------------------------------------------
    # Factory closures wired into ``select_provider`` candidate_factories.
    # Each factory returns a Provider whose ``.load() -> bool`` is what the
    # selector evaluates. The factories preserve the existing
    # ``_try_directml`` / ``_try_ultralytics`` method bodies byte-for-byte
    # by wrapping them in thin provider adapters (Req 5.7).
    # ------------------------------------------------------------------

    def _build_qnn_provider(self, onnx_path: str):
        """Construct a fresh ``QNNProvider`` for the cascade.

        The provider exposes ``.load() -> bool`` and ``.release() -> None`` to
        match the EP_Selector contract.
        """
        from engines.qnn_provider import QNNProvider
        return QNNProvider(
            onnx_path,
            imgsz=self.capture_size,
            conf=self.confidence_threshold,
        )

    def _build_dml_provider(self, onnx_path: str):
        """Construct a Provider that delegates to ``_try_directml``.

        The adapter's ``.load()`` calls the existing ``_try_directml`` method
        unchanged so its body stays byte-for-byte preserved (Req 5.7). It
        translates ``AIEngineException`` to ``False`` so the selector can
        record the failure as a normal cascade entry rather than aborting.
        """
        engine = self

        class _DirectMLAdapter:
            provider_name = "DmlExecutionProvider"

            def load(self) -> bool:
                try:
                    return engine._try_directml(onnx_path)
                except AIEngineException as e:
                    # Selector treats False as "load failed, try next"; the
                    # existing exception-based contract is folded back into a
                    # bool so the cascade can continue.
                    logger.debug("DirectML adapter caught: %s", e)
                    return False

            def release(self) -> None:
                if engine._dml_provider is not None:
                    try:
                        engine._dml_provider.release()
                    except Exception as e:  # noqa: BLE001 — best-effort
                        logger.debug("DirectML adapter release: %s", e)

        return _DirectMLAdapter()

    def _build_ultralytics_provider(self, model_path: str, ext: str):
        """Construct a Provider that delegates to ``_try_ultralytics``.

        Preserves ``_try_ultralytics`` byte-for-byte (Req 5.7).
        """
        engine = self

        class _UltralyticsAdapter:
            provider_name = "Ultralytics"

            def load(self) -> bool:
                try:
                    return engine._try_ultralytics(model_path, ext)
                except AIEngineException as e:
                    logger.debug("Ultralytics adapter caught: %s", e)
                    return False

            def release(self) -> None:
                # Ultralytics teardown happens through engine.release(); no
                # provider-local state to drop here.
                return None

        return _UltralyticsAdapter()

    # ------------------------------------------------------------------
    # Backend state mutation helpers.
    # ``_attach_provider`` and ``_set_backend`` are minimal stubs sufficient
    # for task 6.1; tasks 6.2/6.3 extend them with QNN-specific behavior
    # and the get_health_status / arm64-install-hint logic.
    # ------------------------------------------------------------------

    def _attach_provider(self, provider, backend_name: str) -> None:
        """Store the loaded provider on the right attribute name.

        Mapping:
            "qnn"         -> self._qnn_provider
            "directml"    -> self._dml_provider (already populated by
                             ``_try_directml``; reasserted here for symmetry)
            "ultralytics" -> self._ul_model    (already populated by
                             ``_try_ultralytics``; the adapter is dropped)
        """
        if backend_name == "qnn":
            self._qnn_provider = provider
        # For "directml" and "ultralytics" the underlying ``_try_*`` methods
        # have already populated ``self._dml_provider`` / ``self._ul_model``
        # as a side effect. The adapter wrapper is intentionally not stored.

    def _set_backend(self, backend_name: Optional[str]) -> None:
        """Update ``self._backend`` and shared state ``ai_backend`` together.

        Per Req 5.4a, the shared-state value SHALL strictly equal the active
        backend identifier — one of ``{"qnn", "directml", "ultralytics",
        "none"}``. ``None`` resolves to the literal ``"none"`` so the GUI
        status panel never sees a mismatched pair.

        Any value outside the allowed set is rejected as a programmer error:
        ``self._backend`` is forced to ``None`` and the shared-state value is
        normalized to ``"none"``. This guarantees the engine never reports a
        backend that does not match the actually loaded provider.
        """
        if backend_name is None:
            self._backend = None
            normalized = "none"
        elif backend_name in _VALID_BACKENDS and backend_name != "none":
            # ``"none"`` is reserved for the unloaded state; callers should
            # pass ``None`` to indicate that, not the literal string.
            self._backend = backend_name
            normalized = backend_name
        else:
            logger.warning(
                "_set_backend rejected invalid backend_name=%r; clearing to 'none'",
                backend_name,
            )
            self._backend = None
            normalized = "none"

        if self._shared_state is not None:
            self._shared_state.update_state('ai_backend', normalized)
            self._shared_state.update_state(
                'ai_model_loaded', normalized != "none"
            )

    def _try_directml(self, onnx_path: str) -> bool:
        """Try loading with DirectML provider (AMD GPU)."""
        try:
            from engines.directml_provider import DirectMLProvider, has_directml

            if not has_directml():
                logger.info("DirectML not available, falling back to Ultralytics")
                return False

            self._dml_provider = DirectMLProvider(
                onnx_path,
                imgsz=self.capture_size,
                conf=self.confidence_threshold,
            )

            if not self._dml_provider.load():
                return False

            self._backend = 'directml'
            self.model_loaded = True
            logger.info(f"AI engine ready (DirectML — {self._dml_provider.provider_name})")
            
            # Update state metrics after successful load
            self._update_state_metrics()
            
            return True

        except AIEngineException:
            # Re-raise AIEngineException as-is
            raise
        except Exception as e:
            logger.debug(f"DirectML init failed: {e}")
            if self._shared_state:
                self._shared_state.update_state('ai_engine_error', f"DirectML init failed: {e}")
            raise AIEngineException(f"DirectML backend failed: {e}") from e

    def _try_ultralytics(self, model_path: str, ext: str) -> bool:
        """Fallback to Ultralytics YOLO."""
        try:
            from ultralytics import YOLO
            import torch

            logger.info(f"Loading model via Ultralytics: {model_path} ({ext})")
            self._ul_model = YOLO(model_path)
            self._ul_device = '0' if torch.cuda.is_available() else 'cpu'

            # Warmup
            dummy = np.zeros((self.capture_size, self.capture_size, 3), dtype=np.uint8)
            for _ in range(3):
                self._ul_model.predict(
                    dummy, verbose=False,
                    imgsz=self.capture_size, device=self._ul_device
                )

            self._backend = 'ultralytics'
            self.model_loaded = True
            logger.info(f"AI engine ready (Ultralytics {ext}, device={self._ul_device})")
            
            # Update state metrics after successful load
            self._update_state_metrics()
            
            return True

        except AIEngineException:
            # Re-raise AIEngineException as-is
            raise
        except Exception as e:
            error_msg = f"Ultralytics load failed: {e}"
            logger.error(error_msg)
            if self._shared_state:
                self._shared_state.update_state('ai_engine_error', error_msg)
            raise AIEngineException(error_msg) from e

    def update_config(self, config: dict):
        """Hot-reload configuration without restart.
        
        This method applies configuration changes dynamically, allowing
        the engine to adapt to new settings without requiring a restart.
        Changes are applied immediately and state metrics are updated.
        
        Args:
            config: Dictionary with updated configuration values
        
        Supported config keys:
            - enabled: Enable/disable the engine
            - confidence: Confidence threshold (0.0-1.0)
            - iou_threshold: IOU threshold for NMS (0.0-1.0)
            - capture_size: Capture region size in pixels
            - headshot_bias: Headshot targeting bias (0.0-1.0)
            - target_classes: List of class IDs to detect
        """
        try:
            self.enabled = config.get('enabled', self.enabled)

            # Confidence requires range validation per Req 2.4b: out-of-range
            # values are rejected (WARN) and the previously configured
            # threshold is left in place. The same gate guards propagation to
            # both DirectML and QNN providers — neither should ever receive a
            # value outside ``[0.0, 1.0]``.
            if 'confidence' in config:
                new_conf = config['confidence']
                try:
                    new_conf_f = float(new_conf)
                except (TypeError, ValueError):
                    logger.warning(
                        "update_config rejected confidence=%r (not a number)",
                        new_conf,
                    )
                else:
                    if 0.0 <= new_conf_f <= 1.0:
                        self.confidence_threshold = new_conf_f
                    else:
                        logger.warning(
                            "update_config rejected confidence=%r (out of [0,1])",
                            new_conf,
                        )

            self.iou_threshold = config.get('iou_threshold', self.iou_threshold)
            self.capture_size = config.get('capture_size', self.capture_size)
            self.headshot_bias = config.get('headshot_bias', self.headshot_bias)
            self.target_classes = config.get('target_classes', self.target_classes)

            # Update DirectML conf if active
            if self._dml_provider:
                self._dml_provider.conf = self.confidence_threshold

            # Update QNN conf if active (Reqs 2.4b, 5.5). The provider
            # consumes ``conf`` only at postprocess time, so this is a pure
            # attribute write — no session reload required.
            if self._qnn_provider is not None:
                self._qnn_provider.conf = self.confidence_threshold

            # Update state metrics to reflect new config
            self._update_state_metrics()

            logger.debug(f"AI engine config updated: confidence={self.confidence_threshold}, "
                        f"iou={self.iou_threshold}, enabled={self.enabled}")
        
        except Exception as e:
            logger.error(f"Failed to update AI engine config: {e}", exc_info=True)
            if self._shared_state:
                self._shared_state.update_state('ai_engine_error', str(e))

    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Full inference pipeline on a single frame with error handling.

        Per req 4.9 of the aim-pipeline-simplification spec, this method
        returns the RAW detection list after the class filter and
        confidence threshold — selection / head-bias / sticky-lock now
        live in exactly one place downstream (``aim/pipeline.py::aim_step``
        + ``_select_sticky``). The class filter and confidence threshold
        of req 3.12 / 3.4 are unchanged.

        Args:
            frame: BGR image from capture (capture_size x capture_size)

        Returns:
            List of in-class detections (may be empty). The list is the
            output of the class filter + confidence threshold inside the
            backend-specific ``_process_*`` method; downstream
            ``_select_sticky`` picks the active target.
        """
        # validate_frame skipped in production — capture card always produces
        # valid uint8 HxWx3 frames; the check cost (~5µs) is pure overhead.
        
        if not self.enabled or not self.model_loaded:
            # Update state to reflect no detection
            if self._shared_state:
                self._shared_state.update_state('ai_target_detected', False)
                self._shared_state.update_state('ai_target_class', None)
                self._shared_state.update_state('ai_target_confidence', 0.0)
                self._shared_state.update_state('ai_target_bbox', None)
            return []

        try:
            if self._backend == 'qnn':
                detections = self._process_qnn(frame)
            elif self._backend == 'directml':
                detections = self._process_directml(frame)
            else:
                detections = self._process_ultralytics(frame)
            
            # Update state metrics with detection results
            self._update_detection_state(detections)
            
            return detections
            
        except AIEngineException as e:
            logger.debug(f"AI inference failed: {e}", exc_info=True)
            
            # Update state with error information
            if self._shared_state:
                self._shared_state.update_state('ai_engine_error', str(e))
                self._shared_state.update_state('ai_target_detected', False)
            
            raise
        except Exception as e:
            logger.debug(f"Unexpected error in AI inference: {e}", exc_info=True)
            
            # Update state with error information
            if self._shared_state:
                self._shared_state.update_state('ai_engine_error', str(e))
                self._shared_state.update_state('ai_target_detected', False)
            
            raise AIEngineException(f"Process frame failed: {e}") from e

    def _process_directml(self, frame: np.ndarray) -> List[Detection]:
        """Inference via DirectML ONNX Runtime.

        Returns the raw in-class detection list after the class filter
        + confidence threshold. Selection / head-bias / sticky-lock are
        the responsibility of ``aim_step._select_sticky`` downstream
        (req 4.9). The class filter (``class_id ∈ target_classes``,
        req 3.12) and confidence threshold (req 3.4) are unchanged.
        """
        try:
            raw_dets, infer_ms = self._dml_provider.infer(frame)
            self._record_time(infer_ms)
            self.last_raw_detections = raw_dets or []

            if not raw_dets:
                return []

            # Filter to target classes only and convert to Detection
            detections: List[Detection] = []
            for d in raw_dets:
                if d['class_id'] in self.target_classes:
                    detections.append(Detection(
                        class_id=d['class_id'],
                        class_name=CLASS_NAMES.get(d['class_id'], 'unknown'),
                        x=d['x'], y=d['y'], w=d['w'], h=d['h'],
                        confidence=d['confidence'],
                    ))

            return detections
        
        except Exception as e:
            raise AIEngineException(f"DirectML inference failed: {e}") from e

    def _process_qnn(self, frame: np.ndarray) -> List[Detection]:
        """Inference via QNN ONNX Runtime (Hexagon NPU).

        Mirrors :meth:`_process_directml` byte-for-byte modulo the provider
        attribute it consults: calls ``self._qnn_provider.infer(frame)``,
        records ``infer_ms``, filters detections by ``target_classes``,
        builds ``Detection`` dataclasses with the same field mapping.
        Per req 4.9 (aim-pipeline-simplification), the method now returns
        the raw in-class detection list rather than a single
        closest-to-crosshair selection — selection moves into
        ``aim_step._select_sticky`` downstream. The QNN provider
        cascade and ``Detection`` shape (req 3.4) are unchanged. Per
        Req 6.4 (npu-qnn-provider), any exception raised by the
        provider is re-wrapped in ``AIEngineException`` with the prefix
        ``"QNN inference failed: "`` to match the existing DirectML
        error path's shape.
        """
        try:
            raw_dets, infer_ms = self._qnn_provider.infer(frame)
            self._record_time(infer_ms)
            self.last_raw_detections = raw_dets or []

            if not raw_dets:
                return []

            # Filter to target classes only and convert to Detection
            detections: List[Detection] = []
            for d in raw_dets:
                if d['class_id'] in self.target_classes:
                    detections.append(Detection(
                        class_id=d['class_id'],
                        class_name=CLASS_NAMES.get(d['class_id'], 'unknown'),
                        x=d['x'], y=d['y'], w=d['w'], h=d['h'],
                        confidence=d['confidence'],
                    ))

            return detections

        except Exception as e:
            raise AIEngineException(f"QNN inference failed: {e}") from e

    def _process_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        """Inference via Ultralytics YOLO.

        Returns the raw in-class detection list after the class filter
        + confidence threshold (the ``classes=self.target_classes``
        kwarg already pre-filters at the Ultralytics level). Selection
        is the responsibility of ``aim_step._select_sticky`` (req 4.9).
        """
        try:
            t0 = time.perf_counter()

            results = self._ul_model.predict(
                frame, verbose=False,
                imgsz=self.capture_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                max_det=10,
                device=self._ul_device,
            )

            self._record_time((time.perf_counter() - t0) * 1000)

            if not results or len(results) == 0:
                self.last_raw_detections = []
                return []

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                self.last_raw_detections = []
                return []

            # Parse boxes
            detections: List[Detection] = []
            raw_dets = []
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                x_c = (x1 + x2) / 2
                y_c = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                det = Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, 'unknown'),
                    x=x_c, y=y_c,
                    w=w, h=h,
                    confidence=conf,
                )
                detections.append(det)
                raw_dets.append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "x": x_c, "y": y_c,
                    "w": w, "h": h
                })

            self.last_raw_detections = raw_dets
            return detections
        
        except Exception as e:
            raise AIEngineException(f"Ultralytics inference failed: {e}") from e

    def _record_time(self, ms: float):
        """Record inference time (rolling 100)."""
        if not self._shared_state:
            return  # Skip tracking when no GUI is attached
        self._inference_times.append(ms)
        
        # Update state with latest inference time
        if self._shared_state:
            self._shared_state.update_state('ai_inference_ms', self.get_avg_inference_ms())
    
    def _update_state_metrics(self):
        """Update shared state with current engine metrics.
        
        This method pushes current engine state to the shared state manager,
        allowing the GUI to display real-time metrics. Called after config
        updates and during initialization.
        """
        if not self._shared_state:
            return
        
        try:
            self._shared_state.update_state('ai_backend', self.backend_name)
            self._shared_state.update_state('ai_inference_ms', self.get_avg_inference_ms())
            self._shared_state.update_state('ai_enabled', self.enabled)
            self._shared_state.update_state('ai_model_loaded', self.model_loaded)
        except Exception as e:
            logger.warning(f"Failed to update AI engine state metrics: {e}")
    
    def _update_detection_state(self, detections: List[Detection]):
        """Update shared state with detection results.

        This method pushes detection results to the shared state manager,
        allowing the GUI to display target information in real-time. Per
        req 4.9, the engine no longer performs target selection — selection
        moves to ``aim_step._select_sticky``. For the GUI status panel we
        surface the FIRST detection in the list as the representative
        "current target" (the list is already filtered to in-class
        detections, so any element is a valid target).

        Args:
            detections: List of in-class detections (may be empty).
        """
        if not self._shared_state:
            return

        try:
            if detections:
                detection = detections[0]
                self._shared_state.update_state('ai_target_detected', True)
                self._shared_state.update_state('ai_target_class', detection.class_name)
                self._shared_state.update_state('ai_target_confidence', detection.confidence)
                self._shared_state.update_state('ai_target_bbox',
                    (detection.x, detection.y, detection.w, detection.h))
            else:
                self._shared_state.update_state('ai_target_detected', False)
                self._shared_state.update_state('ai_target_class', None)
                self._shared_state.update_state('ai_target_confidence', 0.0)
                self._shared_state.update_state('ai_target_bbox', None)
        except Exception as e:
            logger.warning(f"Failed to update AI detection state: {e}")

    def get_avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    @property
    def backend_name(self) -> str:
        return self._backend or 'none'

    def get_health_status(self) -> dict:
        """Get engine health status for monitoring and auto-disable logic.
        
        Returns:
            Dictionary with health metrics:
            - operational: bool - whether engine is functioning
            - model_loaded: bool - whether model is loaded
            - backend: str - active backend name
            - avg_inference_ms: float - average inference time
            - enabled: bool - whether engine is enabled

        When the active backend is QNN (``self._backend == "qnn"``), the dict
        is extended with two QNN-specific keys per Req 5.3:
            - execution_provider: "QNNExecutionProvider"
            - htp_performance_mode: "burst"
        """
        avg_inference = 0.0
        if self._inference_times:
            avg_inference = sum(self._inference_times) / len(self._inference_times)

        status = {
            'operational': self.model_loaded and self.enabled,
            'model_loaded': self.model_loaded,
            'backend': self._backend or 'none',
            'avg_inference_ms': avg_inference,
            'enabled': self.enabled
        }

        if self._backend == "qnn":
            status['execution_provider'] = "QNNExecutionProvider"
            status['htp_performance_mode'] = "burst"

        return status

    def release(self):
        """Release all resources and clear state."""
        # QNN provider release (Req 5.6). Always call ``release()`` if a
        # provider instance exists — the provider's own ``release()`` is
        # idempotent and best-effort (Reqs 1.8, 1.8b, 8.4), so it is safe
        # to invoke regardless of whether ``load()`` actually succeeded.
        if self._qnn_provider is not None:
            try:
                self._qnn_provider.release()
            except Exception as e:  # noqa: BLE001 — release must not raise
                logger.warning("QNN provider release raised: %s", e)
            self._qnn_provider = None

        if self._dml_provider:
            self._dml_provider.release()
        self._ul_model = None
        self._dml_provider = None
        self.model_loaded = False
        self._inference_times.clear()

        # Per Req 5.4a / 5.6: backend is cleared to ``None`` (which writes
        # ``shared_state['ai_backend'] = 'none'``) after every release.
        self._set_backend(None)

        # Clear remaining state metrics (target/inference stats) the
        # ``_set_backend(None)`` path does not already cover.
        if self._shared_state:
            self._shared_state.update_state('ai_target_detected', False)
            self._shared_state.update_state('ai_inference_ms', 0.0)

        logger.info("AI engine released")


__all__ = ['AIVisionEngine', 'Detection']
