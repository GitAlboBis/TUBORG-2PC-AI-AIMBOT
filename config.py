"""
Configuration management for the detection framework.

Post-refactoring (single-config-streamlining spec) this module supports only the
Target_Configuration:

    general.architecture = dual_pc
    capture.backend      = capture_card
    general.primary_engine = ai
    input.driver         = kmbox_net

The loader:
  - reads ``config.yaml`` (or raises :class:`ConfigException`);
  - applies default target values for the 4 keys above when absent;
  - delegates value validation to :func:`utils.validation.validate_target_configuration`;
  - delegates ``input.kmbox_net.*`` validation to
    :func:`utils.validation.validate_kmbox_net_config` immediately after the
    Target_Configuration check, so a malformed or missing key fails
    ``load_config()`` with a :class:`ConfigException` that names the offending
    dotted key (Req 3.1, 3.7-3.11);
  - emits diagnostic warnings for every legacy key still present in the file
    (``hsv_engine``, ``memory_esp``, ``input.ib*``, ``input.kmbox_serial*``,
    ``input.makcu_serial*``, ``input.makcu_socket*``, ``input.efi*``,
    ``general.exe_spoof``, and anything containing ``spoof``, ``antidbg`` or
    ``threat_response``);
  - NEVER writes to ``config.yaml`` (Req 2.6, 3.11).
"""

import os
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from exceptions import ConfigException
from utils.logger import setup_logger

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

_logger = setup_logger("config", level="WARNING")

# ---------------------------------------------------------------------------
# Target configuration — single source of truth for accepted values.
# ---------------------------------------------------------------------------

TARGET_CONFIGURATION: Dict[str, str] = {
    "general.architecture": "dual_pc",
    "capture.backend": "capture_card",
    "general.primary_engine": "ai",
    "input.driver": "kmbox_net",
}

# Top-level legacy sections that must not appear in a post-refactoring config.
_LEGACY_TOP_LEVEL_KEYS = ("hsv_engine", "memory_esp")

# Legacy sub-keys under ``input`` (Req 3.9, 7.7).
_LEGACY_INPUT_SUBKEYS = (
    "ib",
    "kmbox_serial",
    "makcu_serial",
    "makcu_socket",
    "efi",
)

# Legacy sub-keys under ``general`` (Req 5.11, 7.7).
_LEGACY_GENERAL_SUBKEYS = ("exe_spoof",)

# Any dotted-path component containing one of these substrings is legacy
# (Req 5.11, 7.7).
_LEGACY_SUBSTRINGS = ("spoof", "antidbg", "threat_response")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load and validate ``config.yaml``.

    Returns:
        The configuration dictionary after default-application and validation.

    Raises:
        ConfigException: ``config.yaml`` is missing, unreadable, malformed, or
            contains a non-target value for any key in
            :data:`TARGET_CONFIGURATION`.
    """
    if not os.path.exists(_CONFIG_FILE):
        raise ConfigException(f"config.yaml not found at {_CONFIG_FILE}")

    try:
        with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigException(f"failed to parse config.yaml: {e}") from e
    except OSError as e:
        raise ConfigException(f"failed to read config.yaml: {e}") from e

    config: Dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    # Warn about every legacy key before applying defaults or validating so the
    # user sees diagnostics even when the loader later raises on a non-target
    # value.
    for legacy_path in _detect_legacy_keys(config):
        _logger.warning("ignored legacy config key: %s", legacy_path)

    _apply_target_defaults(config)
    _validate_target_values(config)
    _normalize_execution_provider(config)
    _validate_aim_and_override(config)
    _log_pixel_to_count(config)

    return config


def reload_config() -> Dict[str, Any]:
    """Reload the configuration from disk.

    Semantically equivalent to :func:`load_config`. Kept as a separate symbol
    so callers can signal intent (hot-reload vs initial load) without any
    runtime write-back to ``config.yaml``.
    """
    return load_config()


# ---------------------------------------------------------------------------
# Convenience accessors (used by the Target_Configuration runtime paths).
# ---------------------------------------------------------------------------

def get_capture_config() -> Dict[str, Any]:
    """Return the ``capture`` section of the loaded configuration."""
    return load_config().get('capture', {})


def get_ai_engine_config() -> Dict[str, Any]:
    """Return the ``ai_engine`` section of the loaded configuration."""
    return load_config().get('ai_engine', {})


def get_input_config() -> Dict[str, Any]:
    """Return the ``input`` section of the loaded configuration."""
    return load_config().get('input', {})


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _apply_target_defaults(config: Dict[str, Any]) -> None:
    """Fill in the 4 Target_Configuration keys with their target default when
    absent.

    Mutates ``config`` in place. Existing values — including non-target ones —
    are preserved so that :func:`_validate_target_values` can surface a precise
    diagnostic citing the offending value.
    """
    for dotted_key, default_value in TARGET_CONFIGURATION.items():
        if not _has_dotted_key(config, dotted_key):
            _set_dotted_key(config, dotted_key, default_value)


def _validate_target_values(config: Dict[str, Any]) -> None:
    """Enforce that each Target_Configuration key holds its target value AND
    that the ``input.kmbox_net.*`` block is well-formed.

    Delegates to :func:`utils.validation.validate_target_configuration` when
    available (installed by task 10.1 of the spec). If the validator is not
    yet present, falls back to an equivalent inline check so the loader still
    honours Req 2.8 / 3.11 / 4.14 / 7.8.

    Immediately after the Target_Configuration check passes, delegates to
    :func:`utils.validation.validate_kmbox_net_config` so a malformed or
    missing ``input.kmbox_net.*`` key fails :func:`load_config` with a
    :class:`ConfigException` that names the offending dotted key (Req 3.1,
    3.7-3.11). No defaults are applied for absent ``input.kmbox_net.*``
    keys: a missing key is a hard error per Req 3.7-3.10.
    """
    try:
        from utils.validation import validate_target_configuration  # lazy import
    except (ImportError, AttributeError):
        validate_target_configuration = None  # type: ignore[assignment]

    if validate_target_configuration is not None:
        validate_target_configuration(config)
    else:
        _logger.warning(
            "utils.validation.validate_target_configuration not available; "
            "using inline fallback validation"
        )
        for dotted_key, target_value in TARGET_CONFIGURATION.items():
            actual = _get_dotted_key(config, dotted_key)
            if actual != target_value:
                raise ConfigException(
                    f"{dotted_key}={actual!r}: unsupported, "
                    f"expected {target_value!r}"
                )

    # Validate the input.kmbox_net.* block. A failure here raises
    # ConfigException naming the offending dotted key, which propagates out of
    # load_config(). Per Req 3.7-3.10 we do NOT default-fill absent keys.
    try:
        from utils.validation import validate_kmbox_net_config  # lazy import
    except (ImportError, AttributeError):
        validate_kmbox_net_config = None  # type: ignore[assignment]

    if validate_kmbox_net_config is None:
        # The validator is part of the same module as
        # validate_target_configuration; if it cannot be imported we cannot
        # honour Req 3.7-3.10 silently — fail loudly so misconfiguration is
        # never masked.
        raise ConfigException(
            "utils.validation.validate_kmbox_net_config is not available; "
            "cannot validate input.kmbox_net.* block"
        )

    validate_kmbox_net_config(config)


def _validate_aim_and_override(config: Dict[str, Any]) -> None:
    """Validate the ``aim.*`` and ``operator_override.*`` blocks.

    Delegates to :func:`utils.validation.validate_aim_config` and
    :func:`utils.validation.validate_operator_override_config`. A
    missing key or a non-positive value raises :class:`ConfigException`
    naming the offending dotted key path (per
    aim-pipeline-simplification req 2.8 / 2.9 / 2.12).

    Spec defaults for the new aim and operator_override sections are
    applied in place when absent — mirroring
    :func:`_apply_target_defaults` for the four Target_Configuration
    keys — so a minimal config (e.g. the kmbox-net-integration
    preservation tests) still loads. Once the defaults are settled the
    validators below are fed a complete dict, and any explicit
    non-positive value the user wrote into ``config.yaml`` raises
    :class:`ConfigException`.

    Runs AFTER :func:`_validate_target_values` so the four
    Target_Configuration keys + four ``input.kmbox_net.*`` keys are
    settled before the additive ``aim.*`` / ``operator_override.*``
    validation fires. The order matters for preservation property P2.5
    (req 3.10): the existing rejection set MUST surface first when a
    config is broken in multiple ways.
    """
    from utils.validation import (  # lazy import
        validate_aim_config,
        validate_operator_override_config,
    )

    _apply_aim_and_override_defaults(config)

    validate_aim_config(config)
    validate_operator_override_config(config)


# Defaults for the new aim.* and operator_override.* blocks added by
# aim-pipeline-simplification task 3.7. The values match the
# canonical ``config.yaml`` shipped in this repo:
#   * ``aim.pixel_to_count = 0.85`` — hipfire scaling for sens 0.5,
#     800 DPI, ADS 0.4 (req 2.8 default)
#   * ``aim.lock_radius_px = 70.0`` — sticky-lock radius (req 2.3)
#   * ``aim.lock_timeout_s = 0.50`` — sticky-lock timeout (req 2.3)
#   * ``aim.max_fov_radius = 200.0`` — FOV-radius filter (req 2.3, 3.12)
#   * ``operator_override.enabled = True`` — gate enabled by default
#   * ``operator_override.threshold_counts = 5`` — req 2.9 default
#   * ``operator_override.window_ms = 50`` — req 2.9 default
_AIM_OVERRIDE_DEFAULTS: Dict[str, Any] = {
    "aim.pixel_to_count": 0.85,
    "aim.lock_radius_px": 70.0,
    "aim.lock_timeout_s": 0.50,
    "aim.max_fov_radius": 200.0,
    "operator_override.enabled": True,
    "operator_override.threshold_counts": 5,
    "operator_override.window_ms": 50,
}


def _apply_aim_and_override_defaults(config: Dict[str, Any]) -> None:
    """Fill in absent ``aim.*`` and ``operator_override.*`` keys with
    their canonical defaults.

    Mutates ``config`` in place. Existing values are preserved so that
    :func:`_validate_aim_and_override` can still surface a precise
    diagnostic when an explicit value in ``config.yaml`` is non-positive
    or wrong-typed.
    """
    for dotted_key, default_value in _AIM_OVERRIDE_DEFAULTS.items():
        if not _has_dotted_key(config, dotted_key):
            _set_dotted_key(config, dotted_key, default_value)


def _log_pixel_to_count(config: Dict[str, Any]) -> None:
    """Emit the ``aim.pixel_to_count`` value at INFO on every config load.

    Required by aim-pipeline-simplification req 2.8: the explicit
    pixel→HID-count scaling factor MUST be visible in logs at INFO
    level on every config load so the operator can trace any
    sensitivity / DPI mismatch directly from the framework's first
    output line.

    The dedicated child logger ``"config.aim"`` is used so this single
    line does not fight the global module logger's WARNING default
    (configured in :func:`utils.logger.setup_logger`); the child is
    pinned to INFO and given its own console handler.
    """
    aim_section = config.get("aim", {})
    pixel_to_count = aim_section.get("pixel_to_count")
    if pixel_to_count is None:  # defensive: validated above
        return

    info_logger = setup_logger("config.aim", level="INFO")
    info_logger.info(
        "aim.pixel_to_count=%.4f (capture-px -> HID-count scaling, req 2.8)",
        float(pixel_to_count),
    )


def _normalize_execution_provider(config: Dict[str, Any]) -> None:
    """Normalize ``ai_engine.execution_provider`` in place.

    Runs after :func:`_validate_target_values` so the Target_Configuration
    keys are settled before this additive npu-qnn-provider key is touched
    (npu-qnn-provider Req 4.1, 4.2, 4.6, 4.8, 4.9).

    Behaviour:
      * When the key is absent, default-fill it with ``"auto"`` (Req 4.2).
      * When the key is present, hand the raw value to
        :func:`utils.validation.validate_execution_provider_value` and write
        the canonical lowercase result back. Unknown string values emit a
        WARN inside the validator and normalize to ``"auto"`` (Req 4.6).
      * ``ConfigException`` raised by the validator (e.g. a list or dict at
        the key, Req 4.9) is left to propagate so :func:`load_config` fails
        with a clear message.

    The ``ai_engine`` section is created if missing — purely defensive; the
    Target_Configuration pass already requires it implicitly through the
    ``general.primary_engine`` default — so the validator always has a
    mapping to write into.
    """
    from utils.validation import validate_execution_provider_value  # lazy import

    ai_engine_section = config.get("ai_engine")
    if not isinstance(ai_engine_section, dict):
        ai_engine_section = {}
        config["ai_engine"] = ai_engine_section

    if "execution_provider" not in ai_engine_section:
        ai_engine_section["execution_provider"] = "auto"
        return

    # Surface ConfigException from the validator unchanged (Req 4.9).
    raw = ai_engine_section["execution_provider"]
    ai_engine_section["execution_provider"] = validate_execution_provider_value(raw)


def _detect_legacy_keys(config: Dict[str, Any]) -> List[str]:
    """Return the dotted paths of every legacy key present in ``config``.

    Detection covers:
      * top-level legacy sections (``hsv_engine``, ``memory_esp``);
      * legacy sub-sections under ``input`` and ``general``;
      * any path whose final component contains ``spoof``, ``antidbg`` or
        ``threat_response``.

    The list is deduplicated while preserving first-seen order so the caller
    can emit diagnostics in a stable sequence.
    """
    found: List[str] = []
    seen: set = set()

    def _record(path: str) -> None:
        if path not in seen:
            seen.add(path)
            found.append(path)

    for top_key in _LEGACY_TOP_LEVEL_KEYS:
        if top_key in config:
            _record(top_key)

    input_section = config.get('input')
    if isinstance(input_section, dict):
        for sub in _LEGACY_INPUT_SUBKEYS:
            if sub in input_section:
                _record(f"input.{sub}")

    general_section = config.get('general')
    if isinstance(general_section, dict):
        for sub in _LEGACY_GENERAL_SUBKEYS:
            if sub in general_section:
                _record(f"general.{sub}")

    for path, _value in _walk(config):
        leaf = path.rsplit('.', 1)[-1].lower()
        if any(bad in leaf for bad in _LEGACY_SUBSTRINGS):
            _record(path)

    return found


# ---------------------------------------------------------------------------
# Dotted-path helpers
# ---------------------------------------------------------------------------

def _split_dotted(dotted_key: str) -> List[str]:
    return dotted_key.split('.')


def _has_dotted_key(config: Dict[str, Any], dotted_key: str) -> bool:
    node: Any = config
    for part in _split_dotted(dotted_key):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _get_dotted_key(config: Dict[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for part in _split_dotted(dotted_key):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_dotted_key(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = _split_dotted(dotted_key)
    node: Dict[str, Any] = config
    for part in parts[:-1]:
        existing = node.get(part)
        if not isinstance(existing, dict):
            existing = {}
            node[part] = existing
        node = existing
    node[parts[-1]] = value


def _walk(node: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every nested mapping entry."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            if isinstance(value, dict):
                yield from _walk(value, path)
