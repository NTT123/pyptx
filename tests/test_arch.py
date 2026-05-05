"""Architecture detection and PTX version mapping for sm_90+ targets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pyptx._arch as arch
from pyptx.kernel import _default_version_for_arch


@pytest.fixture(autouse=True)
def _clear_detect_arch_cache():
    arch.detect_arch.cache_clear()
    yield
    arch.detect_arch.cache_clear()


@pytest.mark.parametrize(
    ("compute_capability", "expected"),
    [
        ((8, 9), "sm_89"),
        ((9, 0), "sm_90a"),
        ((10, 0), "sm_100a"),
        ((10, 3), "sm_103a"),
        ((11, 0), "sm_110a"),
        ((12, 0), "sm_120a"),
        ((12, 1), "sm_121a"),
    ],
)
def test_detect_arch_uses_arch_specific_targets_for_cc_9_and_newer(
    monkeypatch,
    compute_capability,
    expected,
):
    monkeypatch.setattr(arch, "_query_compute_capability", lambda: compute_capability)

    assert arch.detect_arch() == expected


def test_nvidia_smi_compute_capability_fallback(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout="12.0\n")

    monkeypatch.setattr(arch.subprocess, "run", fake_run)

    assert arch._query_compute_capability_from_nvidia_smi() == (12, 0)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("sm_90a", (8, 5)),
        ("sm_100a", (8, 8)),
        ("sm_103a", (8, 8)),
        ("sm_110a", (9, 0)),
        ("sm_120a", (8, 7)),
        ("sm_121a", (8, 8)),
    ],
)
def test_default_ptx_versions_match_auto_selected_targets(target, expected):
    assert _default_version_for_arch(target) == expected
