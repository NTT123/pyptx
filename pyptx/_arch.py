"""Auto-detect the PTX arch string for the first available CUDA device.

Used by ``@kernel(arch="auto")`` and exposed publicly as
``pyptx.detect_arch()`` so portable kernels can be written without
hard-coding a target.

Mapping convention:
  * Compute capability 9.0 (Hopper H100/H200) → ``sm_90a``
  * Compute capability 10.0 (Blackwell datacenter B200/GB200) → ``sm_100a``
  * Compute capability 12.0 (workstation Blackwell, RTX Pro 6000 / RTX 50xx) → ``sm_120a``
    Starting with Compute Capability 9.0, NVIDIA exposes architecture-specific
    feature targets with the ``a`` suffix; using the matching ``sm_120a``
    target unlocks the complete feature set for a 12.0 device.
  * Compute capability <  9.0  → ``sm_{cc}``   (Ampere, Ada, Turing,
    Volta — no arch-specific feature suffix needed).

Detection is best-effort. We try torch first (most users have it via
``pyptx[torch]``), then cuda-python, then ``nvidia-smi``, and as a last
resort raise with a clear message.
"""
from __future__ import annotations

import functools
import subprocess


@functools.lru_cache(maxsize=1)
def detect_arch() -> str:
    """Return the PTX arch string matching the first available CUDA device.

    Examples:
        ``"sm_75"`` (T4),  ``"sm_80"`` (A100),  ``"sm_86"`` (RTX 30xx),
        ``"sm_89"`` (L40 / RTX 40xx),  ``"sm_90a"`` (H100, H200),
        ``"sm_100a"`` (B200, GB200),  ``"sm_120a"`` (RTX Pro 6000
        Blackwell, RTX 50xx).

    Cached after first call (the GPU on a process doesn't change).

    Raises:
        RuntimeError: if no CUDA device is reachable.
    """
    cc = _query_compute_capability()
    if cc is None:
        raise RuntimeError(
            "pyptx.detect_arch(): no CUDA device found. Either install "
            "torch / jax with CUDA support, or pass an explicit arch like "
            "@kernel(arch='sm_80')."
        )
    major, minor = cc
    cap = major * 10 + minor
    # CC 9.0+ devices have architecture-specific feature targets. The
    # matching ``a`` target is the right default for code generated for
    # the exact local GPU; users can still request plain/family targets
    # explicitly when they need wider binary compatibility.
    if cap in (90, 100, 101, 103, 110, 120, 121):
        return f"sm_{cap}a"
    return f"sm_{cap}"


def _query_compute_capability() -> tuple[int, int] | None:
    """Try torch, cuda-python, then nvidia-smi. Returns (major, minor) or None."""
    # torch path — fastest, most users have it.
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass

    # cuda-python path — standalone fallback.
    try:
        try:
            from cuda.bindings import driver as cuda  # cuda-python >= 12.8
        except ImportError:
            from cuda import cuda  # type: ignore[no-redef]  # older shim
        cuda.cuInit(0)
        _, dev = cuda.cuDeviceGet(0)
        attr_major = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        attr_minor = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        _, major = cuda.cuDeviceGetAttribute(attr_major, dev)
        _, minor = cuda.cuDeviceGetAttribute(attr_minor, dev)
        return (int(major), int(minor))
    except Exception:
        pass

    cc = _query_compute_capability_from_nvidia_smi()
    if cc is not None:
        return cc

    return None


def _query_compute_capability_from_nvidia_smi() -> tuple[int, int] | None:
    """Return the first GPU compute capability reported by nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            major_s, minor_s = value.split(".", 1)
            return int(major_s), int(minor_s)
        except Exception:
            return None
    return None
