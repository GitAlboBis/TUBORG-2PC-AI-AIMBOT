"""
QNN ONNX Inference Provider — Qualcomm Hexagon NPU acceleration for YOLO models.

This module provides a direct ONNX Runtime session with QNNExecutionProvider,
targeting the Hexagon Tensor Processor (HTP) backend on Snapdragon X (ARM64) hosts.

Design contract mirrors `engines/directml_provider.py`:
    provider = QNNProvider("models/v11n-416-2.onnx", imgsz=416, conf=0.55)
    if provider.load():
        detections, ms = provider.infer(bgr_frame)
        provider.release()

Requires: pip install onnxruntime-qnn  (cannot coexist with onnxruntime-directml).

Notes:
- `onnxruntime` is NOT imported at module load (Req 7.5). Importing this module on a
  host without ONNX Runtime is non-fatal; lazy imports happen inside `load()` and inside
  the diagnostics helpers.
- The lifecycle methods `preprocess`, `postprocess`, `infer`, and `release` are
  fully implemented here; `__init__`, `load`, and the module helpers (`has_qnn`,
  `get_qnn_diagnostics`) are operational alongside them.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# Candidate basenames for the QNN HTP backend shared library across platforms.
_HTP_LIB_CANDIDATES = ("QnnHtp.dll", "libQnnHtp.so")


def _resolve_htp_backend_path() -> Optional[str]:
    """
    Locate the QNN HTP backend library.

    Resolution order (per design §load step 4):
      1. ``$QNN_HTP_BACKEND_PATH`` environment variable (explicit override).
      2. ``onnxruntime_qnn.get_qnn_htp_path()`` — official query helper shipped
         by the ``onnxruntime-qnn`` plugin package; resolves to the correct
         ARM64EC / AMD64 directory automatically.
      3. ``<sys.prefix>/Lib/site-packages/onnxruntime/capi/QnnHtp.dll`` — legacy
         install location for older ORT-bundled QNN builds on Windows.
      4. First ``QnnHtp.dll`` (or ``libQnnHtp.so``) found on ``PATH``.

    Returns:
        Absolute path to the resolved library, or ``None`` if no candidate exists.
    """
    # 1) Explicit override.
    env_override = os.environ.get("QNN_HTP_BACKEND_PATH")
    if env_override and os.path.exists(env_override):
        return env_override

    # 2) onnxruntime-qnn plugin helper — preferred since it picks the correct
    #    arch-specific subdirectory (arm64ec on Snapdragon X, amd64 on x86_64
    #    cross-compiles, etc).
    try:
        import onnxruntime_qnn  # type: ignore[import-not-found]

        candidate = onnxruntime_qnn.get_qnn_htp_path()
        if candidate and os.path.exists(candidate):
            return candidate
    except Exception:  # noqa: BLE001 — onnxruntime_qnn may not be installed
        pass

    # 3) Legacy ORT-bundled QNN install location (older builds shipped the DLLs
    #    under <prefix>/Lib/site-packages/onnxruntime/capi).
    #    On Windows the DLL lives under <prefix>/Lib/site-packages/onnxruntime/capi.
    #    On Linux the .so lives under <prefix>/lib/python*/site-packages/onnxruntime/capi.
    capi_candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime", "capi"),
        os.path.join(
            sys.prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "onnxruntime",
            "capi",
        ),
    ]
    # Also probe the actual onnxruntime install dir if importable, so virtualenvs
    # with non-standard layouts still resolve.
    try:
        import onnxruntime as _ort_for_path

        ort_dir = os.path.dirname(getattr(_ort_for_path, "__file__", "") or "")
        if ort_dir:
            capi_candidates.append(os.path.join(ort_dir, "capi"))
    except Exception:  # noqa: BLE001 — best-effort directory discovery
        pass

    for capi_dir in capi_candidates:
        for lib_name in _HTP_LIB_CANDIDATES:
            candidate = os.path.join(capi_dir, lib_name)
            if os.path.exists(candidate):
                return candidate

    # 3) Walk PATH for the bare basename.
    path_env = os.environ.get("PATH", "")
    for path_dir in path_env.split(os.pathsep):
        if not path_dir:
            continue
        for lib_name in _HTP_LIB_CANDIDATES:
            candidate = os.path.join(path_dir, lib_name)
            if os.path.exists(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Module-level helpers (Req 7.1, 7.2)
# ---------------------------------------------------------------------------

def _normalize_host_arch() -> str:
    """Normalize ``platform.machine()`` to one of ``arm64`` / ``x86_64`` / ``other``."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("amd64", "x86_64"):
        return "x86_64"
    return "other"


def _ensure_qnn_registered() -> bool:
    """
    Register the ``onnxruntime-qnn`` plugin EP with ONNX Runtime if needed.

    ORT 1.20+ ships QNN as an out-of-tree plugin EP rather than a built-in
    provider. ``onnxruntime-qnn`` only ships the QNN backend DLLs; the actual
    registration with the host ``onnxruntime`` Python module must happen via
    ``onnxruntime.register_execution_provider_library``. We do that exactly
    once per process, idempotently — repeated registration is a no-op.

    Returns ``True`` when ``"QNNExecutionProvider"`` is in the list of
    available providers after the call (registration succeeded or was already
    in effect); ``False`` otherwise.
    """
    try:
        import onnxruntime as ort  # local import — Req 7.5
    except ImportError:
        return False

    # Already registered (either by us in a previous call or built into the
    # ORT wheel itself).
    try:
        if "QNNExecutionProvider" in ort.get_available_providers():
            return True
    except Exception:  # noqa: BLE001
        return False

    # The plugin loader lives in ``onnxruntime_qnn``; if it's not installed
    # there's nothing to register.
    try:
        import onnxruntime_qnn  # type: ignore[import-not-found]
    except ImportError:
        return False

    register = getattr(ort, "register_execution_provider_library", None)
    if register is None:
        # Older ORT without plugin-EP support — caller will fall through to
        # the install-hint path.
        return False

    try:
        lib_path = onnxruntime_qnn.get_library_path()
    except Exception as e:  # noqa: BLE001
        logger.error("onnxruntime_qnn.get_library_path() failed: %s", e)
        return False
    if not os.path.exists(lib_path):
        logger.error("QNN provider library missing at %s", lib_path)
        return False

    try:
        register("QNNExecutionProvider", lib_path)
    except Exception as e:  # noqa: BLE001
        logger.error("register_execution_provider_library('QNN', %s): %s", lib_path, e)
        return False

    try:
        return "QNNExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def has_qnn() -> bool:
    """
    Return ``True`` iff ``QNNExecutionProvider`` is registered with ONNX Runtime
    (Req 7.1).

    Triggers the once-per-process plugin registration via
    :func:`_ensure_qnn_registered` so the canonical provider name appears in
    ``onnxruntime.get_available_providers()`` whenever the ``onnxruntime-qnn``
    package is installed alongside a plugin-EP-aware ``onnxruntime`` build.

    Tolerates a missing ``onnxruntime`` install: returns ``False`` instead of raising.
    """
    return _ensure_qnn_registered()


def get_qnn_diagnostics() -> dict:
    """
    Return a structured diagnostics dictionary for QNN environment detection (Req 7.2).

    Keys:
        onnxruntime_installed: bool
        onnxruntime_version:   Optional[str]
        qnn_available:         bool
        available_providers:   list[str]
        host_arch:             str  (``arm64`` / ``x86_64`` / ``other``)

    Tolerates ``ImportError`` so this helper is always safe to call.
    """
    diagnostics = {
        "onnxruntime_installed": False,
        "onnxruntime_version": None,
        "qnn_available": False,
        "available_providers": [],
        "host_arch": _normalize_host_arch(),
    }
    try:
        import onnxruntime as ort  # local import — Req 7.5
    except ImportError:
        return diagnostics

    diagnostics["onnxruntime_installed"] = True
    diagnostics["onnxruntime_version"] = getattr(ort, "__version__", None)
    # Trigger plugin-EP registration if needed so the diagnostics reflect the
    # post-bootstrap state, not the pre-bootstrap one.
    _ensure_qnn_registered()
    try:
        providers = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        providers = []
    diagnostics["available_providers"] = providers
    diagnostics["qnn_available"] = "QNNExecutionProvider" in providers
    return diagnostics


# ---------------------------------------------------------------------------
# QNNProvider
# ---------------------------------------------------------------------------

class QNNProvider:
    """
    ONNX Runtime inference provider backed by the Qualcomm QNN HTP execution provider.

    The class shape mirrors :class:`engines.directml_provider.DirectMLProvider` so that
    ``AIVisionEngine._process_qnn`` can be a near-identical sibling of
    ``AIVisionEngine._process_directml``.
    """

    def __init__(self, model_path: str, imgsz: int = 416, conf: float = 0.55) -> None:
        """
        Args:
            model_path: Filesystem path to the ``.onnx`` model artifact.
            imgsz:      Square model input edge size (default 416 for YOLO11n-416).
            conf:       Confidence threshold; must lie in ``[0.0, 1.0]`` (Req 2.4a).

        Raises:
            ValueError: when ``conf`` is outside ``[0.0, 1.0]`` (Req 2.4a).
        """
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"conf={conf!r} not in [0.0, 1.0]")

        # Constructor arguments
        self.model_path: str = model_path
        self.imgsz: int = imgsz
        self.conf: float = conf

        # ORT session state — populated by load()
        self.session = None  # type: ignore[assignment]
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self.provider_used: str = "none"

        # QNN-specific state — populated by load()
        self._io_binding = None  # type: ignore[assignment]
        self._input_buffer: Optional[np.ndarray] = None
        self._resize_buffer: Optional[np.ndarray] = None
        self._input_ortvalue = None  # type: ignore[assignment]
        self._load_thread_id: Optional[int] = None
        self._htp_lib_path: Optional[str] = None
        # Scalar used by ``preprocess`` to normalize uint8 frames into the
        # bound dtype (FP16 by design, FP32 when the model declares
        # ``tensor(float)``). Set by ``load()`` after probing the input meta.
        self._input_norm_scalar = np.float16(255.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Initialize the QNN ONNX Runtime session against the Hexagon NPU (Req 1.2 – 1.7).

        Returns:
            ``True`` if the session is created with ``QNNExecutionProvider`` and the
            three-inference warmup completed; ``False`` otherwise. Never re-raises
            exceptions raised by ONNX Runtime — those are logged and surfaced as a
            ``False`` return so the surrounding ``EP_Selector`` cascade can fall through
            to the next candidate (Reqs 1.6, 6.2, 6.3).
        """
        # 1) Lazy import — Req 7.5. Importing this module on a host without ORT must
        #    not raise; the install hint is the operator-friendly diagnostic.
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "onnxruntime not installed — install with `pip install onnxruntime-qnn` "
                "to enable the QNN execution provider"
            )
            return False

        # 2) Provider availability — Req 1.2 prereq, Req 6.1.
        # ORT 1.20+ ships QNN as a plugin EP that has to be registered before
        # it appears in ``get_available_providers()``. ``_ensure_qnn_registered``
        # is idempotent — repeated calls are no-ops.
        _ensure_qnn_registered()
        try:
            available_providers = list(ort.get_available_providers())
        except Exception as e:  # noqa: BLE001 — defensive against ORT internals
            logger.info("onnxruntime.get_available_providers() failed: %s", e)
            return False
        if "QNNExecutionProvider" not in available_providers:
            logger.info(
                "QNNExecutionProvider not in available providers (%s); "
                "QNNProvider.load() returning False",
                available_providers,
            )
            return False

        # 3) Model file check — Req 1.4.
        if not os.path.exists(self.model_path):
            logger.error("ONNX model not found at %s", self.model_path)
            return False

        # 4) Resolve HTP backend library path. On miss, log ERROR and return False.
        htp_lib = _resolve_htp_backend_path()
        if htp_lib is None:
            logger.error(
                "QNN HTP backend library not found — checked $QNN_HTP_BACKEND_PATH, "
                "<sys.prefix>/Lib/site-packages/onnxruntime/capi/QnnHtp.dll, and PATH"
            )
            return False
        self._htp_lib_path = htp_lib

        # 5) Provider options — gap-resolution #1, Req 1.2.
        provider_options = {
            "backend_path": self._htp_lib_path,
            "htp_performance_mode": "burst",
            "log_severity_level": "3",  # Suppress native stage timing and DDR summaries
        }
        # ORT 1.20+ ships QNN as an out-of-tree plugin EP. The legacy
        # ``providers=[("QNNExecutionProvider", opts), ...]`` constructor path
        # silently falls back to CPU because the new provider type is not
        # known to the CreateSessionFromArray fast-path. Bind via the new
        # ``SessionOptions.add_provider_for_devices`` API instead, picking the
        # first OrtEpDevice whose ``device.type`` is the Hexagon NPU. We also
        # keep a ``providers=`` arg for legacy ORT (< 1.20) compatibility.
        target_ep_devices = []
        try:
            all_ep_devices = list(ort.get_ep_devices())
        except Exception:  # noqa: BLE001
            all_ep_devices = []
        # Prefer NPU (HTP) devices. ``OrtHardwareDeviceType.NPU`` is the right
        # discriminator on Snapdragon X Elite; falling back to GPU then CPU
        # would mask a QNN-misconfiguration so we deliberately avoid it.
        try:
            from onnxruntime import OrtHardwareDeviceType  # type: ignore[attr-defined]
            npu_type = OrtHardwareDeviceType.NPU
        except Exception:  # noqa: BLE001
            npu_type = None
        for dev in all_ep_devices:
            if dev.ep_name != "QNNExecutionProvider":
                continue
            if npu_type is not None and dev.device.type != npu_type:
                continue
            target_ep_devices.append(dev)

        # 6) Session options — gap-resolution #7.
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        # 6b) QNN context-binary cache (EPContext). The HTP graph
        # finalize/compile dominates session creation (~1.9s for yolo11n,
        # ~7.5s for yolov8m, measured). With a cached context the load drops
        # to ~0.3s. If a fresh cache exists next to the model, load from it;
        # otherwise compile from the source model and ask ORT to dump the
        # context for the next startup. NOTE: these are SessionOptions
        # config entries, NOT provider_options keys.
        ctx_path = os.path.splitext(self.model_path)[0] + "_ctx.onnx"
        session_model_path = self.model_path
        try:
            ctx_is_fresh = (
                os.path.exists(ctx_path)
                and os.path.getmtime(ctx_path) >= os.path.getmtime(self.model_path)
            )
        except OSError:
            ctx_is_fresh = False
        if ctx_is_fresh:
            session_model_path = ctx_path
        else:
            try:
                opts.add_session_config_entry("ep.context_enable", "1")
                opts.add_session_config_entry("ep.context_file_path", ctx_path)
                opts.add_session_config_entry("ep.context_embed_mode", "1")
            except Exception as e:  # noqa: BLE001 — older ORT without EPContext
                logger.info("EPContext cache unavailable (%s); plain load", e)

        # ORT 1.20+ plugin-EP API: register QNN against the discovered NPU
        # device(s) BEFORE constructing the session. ``provider_options`` is a
        # ``dict[str, str]`` here per ``add_provider_for_devices``'s mapping
        # contract.
        plugin_api_used = False
        if target_ep_devices and hasattr(opts, "add_provider_for_devices"):
            try:
                opts.add_provider_for_devices(
                    target_ep_devices,
                    {k: str(v) for k, v in provider_options.items()},
                )
                plugin_api_used = True
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "add_provider_for_devices failed (%s); falling back to legacy providers= path",
                    e,
                )

        # Legacy ORT (< 1.20) fallback: pass ``providers=`` to InferenceSession.
        # Only used when the plugin-EP API path didn't bind the provider above.
        legacy_providers = (
            None
            if plugin_api_used
            else [
                ("QNNExecutionProvider", provider_options),
                "CPUExecutionProvider",
            ]
        )

        # 7) Construct session. Single broad except branch covers Reqs 1.6, 6.2, 6.3 —
        #    we never re-raise out of load(); EP_Selector matches "no NPU detected" on
        #    the logged message text.
        def _make_session(path):
            if legacy_providers is not None:
                return ort.InferenceSession(
                    path,
                    sess_options=opts,
                    providers=legacy_providers,
                )
            # ORT 1.20+ plugin-EP path — providers were already bound via
            # opts.add_provider_for_devices above. Passing ``providers=``
            # again would override that binding.
            return ort.InferenceSession(path, sess_options=opts)

        try:
            session = _make_session(session_model_path)
        except Exception as e:  # noqa: BLE001 — broad-by-design per Req 1.6
            if session_model_path != self.model_path:
                # Stale/incompatible context cache (e.g. QNN SDK update):
                # drop it and recompile from the source model.
                logger.warning(
                    "QNN context cache load failed (%s); recompiling from %s",
                    e, self.model_path,
                )
                try:
                    os.remove(ctx_path)
                except OSError:
                    pass
                try:
                    opts.add_session_config_entry("ep.context_enable", "1")
                    opts.add_session_config_entry("ep.context_file_path", ctx_path)
                    opts.add_session_config_entry("ep.context_embed_mode", "1")
                except Exception:  # noqa: BLE001
                    pass
                try:
                    session = _make_session(self.model_path)
                except Exception as e2:  # noqa: BLE001
                    logger.error("QNN InferenceSession creation failed: %s", e2)
                    self.session = None
                    return False
            else:
                logger.error("QNN InferenceSession creation failed: %s", e)
                self.session = None
                return False

        # 8) Probe I/O metadata.
        try:
            input_meta = session.get_inputs()[0]
            input_name = input_meta.name
            output_names = [o.name for o in session.get_outputs()]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to probe QNN session I/O metadata: %s", e)
            return False

        # 8b) Pick the input dtype to match what the ONNX graph declares.
        # The design defaults to FP16 (QNN HTP's native tensor format) but the
        # ONNX artifact may have been exported in FP32 — using the wrong dtype
        # surfaces as ``Unexpected input data type`` from the ORT session at
        # warmup time. Probe the declared type and keep both branches working.
        declared_type = getattr(input_meta, "type", "tensor(float16)")
        # Coerce to a string defensively so the substring matches behave even
        # when the meta is a Mock (test fixtures use MagicMock for InputMeta).
        try:
            declared_type_str = str(declared_type)
        except Exception:  # noqa: BLE001
            declared_type_str = ""
        if "float16" in declared_type_str:
            input_dtype = np.float16
            input_dtype_norm = np.float16(255.0)
        elif "float" in declared_type_str:  # tensor(float) == FP32
            input_dtype = np.float32
            input_dtype_norm = np.float32(255.0)
        else:
            # No declared type information (e.g. Mock InputMeta) — default to
            # FP16 to keep the design intent. Real ONNX sessions always have
            # a real dtype on get_inputs()[0].type.
            input_dtype = np.float16
            input_dtype_norm = np.float16(255.0)
        # Stash for ``preprocess`` so it can normalize against the right scale.
        self._input_norm_scalar = input_dtype_norm

        # 9) Allocate the pre-allocated NCHW input buffer and the HWC uint8 resize
        #    scratch — gap-resolution #6. No per-frame allocations after this point.
        resize_buffer = np.empty((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        input_buffer = np.empty(
            (1, 3, self.imgsz, self.imgsz), dtype=input_dtype
        )

        # 10) Bind the input OrtValue once. Try QNN device-side allocation first; on
        #     any failure (older ORT builds, unsupported device kwarg) fall back to
        #     pinned-host allocation. Both keep `infer()` allocation-free.
        try:
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                input_buffer, "qnn", 0
            )
        except Exception:  # noqa: BLE001 — fall back to pinned-host allocation
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_buffer)

        try:
            io_binding = session.io_binding()
            io_binding.bind_ortvalue_input(input_name, input_ortvalue)
            # Bind every output to host memory so `copy_outputs_to_cpu()` in `infer()`
            # works without per-call rebinding (gap-resolution #5).
            for out_name in output_names:
                io_binding.bind_output(out_name, "cpu")
        except Exception as e:  # noqa: BLE001
            logger.error("QNN IOBinding setup failed: %s", e)
            return False

        # 11) Three-inference warmup against the zero-filled input buffer — Req 1.7,
        #     gap-resolution #10. Forces HTP graph compilation so the first user-frame
        #     inference latency is steady-state.
        try:
            input_buffer.fill(0)
            # If the OrtValue holds device-side storage, mirror the zero-fill across the
            # binding before each warmup run.
            for _ in range(3):
                try:
                    input_ortvalue.update_inplace(input_buffer)
                except Exception:  # noqa: BLE001 — pinned-host already shares memory
                    pass
                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
        except Exception as e:  # noqa: BLE001
            logger.error("QNN warmup inference failed: %s", e)
            return False

        # 12) Defensive parity guard — verify the session actually picked up
        #     QNNExecutionProvider as the first provider. If it silently fell back to
        #     CPU we want to fail closed rather than mis-report the backend.
        try:
            actual = session.get_providers()[0]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to read session providers post-load: %s", e)
            return False
        if actual != "QNNExecutionProvider":
            logger.warning("expected QNN, got %s", actual)
            return False

        # Commit the session state in one atomic batch. Anything that could fail has
        # already been guarded above.
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        self._resize_buffer = resize_buffer
        self._input_buffer = input_buffer
        self._input_ortvalue = input_ortvalue
        self._io_binding = io_binding
        self._load_thread_id = threading.get_ident()
        self.provider_used = "QNNExecutionProvider"

        logger.info(
            "QNNProvider loaded: model=%s, imgsz=%d, conf=%.2f, htp_backend=%s, "
            "input=%s, outputs=%s",
            self.model_path,
            self.imgsz,
            self.conf,
            self._htp_lib_path,
            self.input_name,
            self.output_names,
        )
        return True

    def preprocess(self, frame: np.ndarray) -> None:
        """
        Write a BGR ``uint8`` frame into the pre-allocated FP16 NCHW input buffer
        in place — no per-frame allocations (Req 2.7, gap-resolution #6).

        Steps (mirrors :class:`DirectMLProvider.preprocess` minus the per-call
        ``np.empty`` plus the cast-to-FP16 demanded by QNN):

            1. ``cv2.resize`` into the pre-allocated ``_resize_buffer`` (HWC uint8).
            2. ``cv2.cvtColor`` BGR → RGB in place into the same buffer.
            3. ``np.divide`` the transposed ``[None, ...]`` view by ``float16(255.0)``
               with ``out=self._input_buffer, casting='unsafe'`` so the FP16 NCHW
               tensor is written without an intermediate allocation.

        The numpy ``transpose`` returns a view with non-contiguous strides, but
        ``np.divide`` accepts strided sources so long as the output (here the
        contiguous ``_input_buffer``) absorbs the values element-wise.

        Args:
            frame: BGR ``uint8`` image of shape ``(H, W, 3)`` from the capture pipeline.
        """

        # Step 1: in-place resize. Some OpenCV builds reject ``dst`` for shape changes,
        # so guard the fast path and fall back to ``np.copyto`` from a freshly resized
        # ndarray (still avoids accumulating allocations because ``np.copyto`` writes
        # into the pre-allocated buffer; the temporary is GC'd immediately).
        try:
            cv2.resize(
                frame,
                (self.imgsz, self.imgsz),
                dst=self._resize_buffer,
                interpolation=cv2.INTER_LINEAR,
            )
        except (cv2.error, TypeError):
            np.copyto(
                self._resize_buffer,
                cv2.resize(
                    frame,
                    (self.imgsz, self.imgsz),
                    interpolation=cv2.INTER_LINEAR,
                ),
            )

        # Step 2: in-place BGR → RGB.
        cv2.cvtColor(self._resize_buffer, cv2.COLOR_BGR2RGB, dst=self._resize_buffer)

        # Step 3: HWC uint8 → NCHW [0.0, 1.0] in a single contiguous write. The
        # ``_input_norm_scalar`` is a scalar in the same dtype as
        # ``_input_buffer`` so ``np.divide`` produces output in the bound dtype
        # (FP16 for the design default, FP32 for FP32-exported models).
        np.divide(
            self._resize_buffer.transpose(2, 0, 1)[None, ...],
            self._input_norm_scalar,
            out=self._input_buffer,
            casting="unsafe",
        )

    def postprocess(
        self, output: np.ndarray, orig_w: int, orig_h: int
    ) -> List[dict]:
        """
        Convert raw YOLO output ``(1, 6, N)`` into a list of ``Detection_Dict`` (Req 2.2,
        2.3, 2.3a, 2.4, 2.5, 2.8). Mirrors :meth:`DirectMLProvider.postprocess` exactly
        in structure; adds the bbox clamp and zero-area drop demanded by Req 2.3a /
        gap-resolution #4.

        Args:
            output:  Raw model output of shape ``(1, 6, N)`` where the 6 channels are
                     ``[cx, cy, w, h, cls0_conf, cls1_conf]`` in model-input pixel space.
            orig_w:  Width of the original capture frame, used for coordinate scaling.
            orig_h:  Height of the original capture frame, used for coordinate scaling.

        Returns:
            A list of ``Detection_Dict`` instances with exactly the six keys
            ``class_id, confidence, x, y, w, h`` (all coerced to native Python
            ``int`` / ``float`` so consumers never see numpy scalars).
        """

        # (1, 6, N) → (N, 6).
        preds = output[0].T

        # Split boxes (cx, cy, w, h) and per-class confidences (Req 2.4).
        boxes = preds[:, :4]
        class_scores = preds[:, 4:]

        # Best class per detection.
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Confidence-threshold mask (Req 2.4).
        mask = confidences >= self.conf
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Centre format → ``(x1, y1, w, h)`` for ``cv2.dnn.NMSBoxes`` (Req 2.5,
        # gap-resolution #8). ``cv2.dnn.NMSBoxes`` accepts Python lists of floats; cast
        # via ``.tolist()`` to also coerce out of FP16 storage.
        x1 = boxes[:, 0] - boxes[:, 2] / 2.0
        y1 = boxes[:, 1] - boxes[:, 3] / 2.0
        nms_boxes = np.stack([x1, y1, boxes[:, 2], boxes[:, 3]], axis=1).astype(
            np.float32, copy=False
        ).tolist()
        nms_scores = confidences.astype(np.float32, copy=False).tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.conf, 0.45)

        if len(indices) == 0:
            return []

        # Independent x/y scale factors (Req 2.8).
        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)
        orig_w_f = float(orig_w)
        orig_h_f = float(orig_h)

        detections: List[dict] = []
        for idx in indices:
            # Older OpenCV bindings return a list of single-element arrays;
            # newer return a flat array. Normalize to a scalar index.
            i = int(idx[0]) if hasattr(idx, "__len__") else int(idx)

            # Centre coordinates and dimensions in original-frame pixels.
            cx = float(boxes[i, 0]) * scale_x
            cy = float(boxes[i, 1]) * scale_y
            bw = float(boxes[i, 2]) * scale_x
            bh = float(boxes[i, 3]) * scale_y

            # Clamp per Req 2.3a / gap-resolution #4. ``min/max`` form chosen so NaN
            # propagates cleanly (np.clip would silently swallow NaN with bounds set).
            cx = max(0.0, min(cx, orig_w_f))
            cy = max(0.0, min(cy, orig_h_f))
            bw = max(0.0, min(bw, orig_w_f))
            bh = max(0.0, min(bh, orig_h_f))

            # Drop zero-area detections silently (Req 2.3a).
            if bw == 0.0 or bh == 0.0:
                continue

            detections.append(
                {
                    "class_id": int(class_ids[i]),
                    "confidence": float(confidences[i]),
                    "x": cx,
                    "y": cy,
                    "w": bw,
                    "h": bh,
                }
            )

        return detections

    def infer(self, frame: np.ndarray) -> Tuple[List[dict], float]:
        """
        Run preprocess → session → postprocess and return ``(detections, elapsed_ms)``
        (Reqs 2.1, 6.4).

        Steps:
            1. Capture ``t0`` and the original frame dimensions.
            2. Write the FP16 NCHW input tensor in place via :meth:`preprocess`.
            3. ``synchronize_inputs()`` — uploads the binding to the HTP DMA channel
               on QNN; a no-op on CPU EP.
            4. ``session.run_with_iobinding(...)`` — the actual NPU inference.
            5. ``copy_outputs_to_cpu()`` — pulls the raw output tensor back to host.
            6. Pass through :meth:`postprocess` to obtain Detection_Dict instances in
               original-frame coordinates.

        Per Req 6.4, any exception raised by preprocess, the ORT session, or
        postprocess is **re-raised unchanged**. The surrounding
        ``AIVisionEngine._process_qnn`` is responsible for wrapping in
        :class:`AIEngineException`.

        Args:
            frame: BGR ``uint8`` image of shape ``(H, W, 3)`` from the capture
                pipeline.

        Returns:
            Tuple of ``(detections, elapsed_ms)`` where ``detections`` is a list of
            Detection_Dict instances and ``elapsed_ms`` is a non-negative ``float``
            measured via :func:`time.perf_counter`.
        """
        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        self.preprocess(frame)
        self._io_binding.synchronize_inputs()
        self.session.run_with_iobinding(self._io_binding)
        outputs = self._io_binding.copy_outputs_to_cpu()
        detections = self.postprocess(outputs[0], orig_w, orig_h)
        return detections, (time.perf_counter() - t0) * 1000.0

    def release(self) -> None:
        """
        Drop the ONNX Runtime session and pre-allocated buffers
        (Reqs 1.8, 1.8a, 1.8b, 8.3, 8.4, gap-resolution #2).

        Behaviour:
            - Logs a WARN when called from a thread other than the one that
              performed ``load()`` (Reqs 1.8b, 8.4); the call still proceeds and
              never raises.
            - Drops references in lifecycle order — IO binding before the session,
              then the input ``OrtValue``, then the host-side numpy buffers — so
              ORT internals see a consistent teardown sequence.
            - Any exception during the drop sequence is logged at WARN and
              swallowed; teardown is best-effort (gap-resolution #2).
            - ``provider_used`` is reset to ``"none"`` in a ``finally`` so the
              public state always reflects "no provider attached" after the call.
            - If the wall-clock elapsed exceeds 2000 ms, a WARN is logged but no
              exception is raised (gap-resolution #2; Req 1.8a soft-bound).
            - Idempotent: a second call after the first (or a call before
              ``load()`` ever ran) is a no-op and never raises (Req 1.8).
        """
        t0 = time.perf_counter()
        cur = threading.get_ident()
        if self._load_thread_id is not None and cur != self._load_thread_id:
            logger.warning(
                "QNNProvider.release() called from thread %d (loaded on %d); "
                "executing best-effort",
                cur,
                self._load_thread_id,
            )
        try:
            # Drop binding before session per ORT lifecycle, then the OrtValue,
            # then the host-side scratch buffers (gap-resolution #2).
            self._io_binding = None
            self._input_ortvalue = None
            self.session = None
            self._input_buffer = None
            self._resize_buffer = None
        except Exception as e:  # noqa: BLE001 — best-effort teardown, never raise
            logger.warning("QNNProvider.release encountered %s; continuing", e)
        finally:
            self.provider_used = "none"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 2000.0:
            logger.warning(
                "QNNProvider.release exceeded 2 s budget: %.1f ms", elapsed_ms
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Return the active ORT provider identifier (``"none"`` until ``load()`` runs)."""
        return self.provider_used


__all__ = ["QNNProvider", "has_qnn", "get_qnn_diagnostics"]
