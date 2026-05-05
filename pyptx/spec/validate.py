"""Spec-driven validation for PTX IR instructions.

Validation is optional and separate from parsing. The parser accepts
anything syntactically valid. This module checks parsed Instruction nodes
against the declarative spec in pyptx.spec.ptx.

The validator supports:

  * Multiple specs (overloads) per opcode. The base table in
    ``pyptx.spec.ptx`` only stores one spec per opcode (last write wins),
    so this module also maintains an overload registry where additional
    specs can be registered. Validation tries every spec for an opcode
    and reports the issues from the best-matching one.
  * Strict mode (the default). When strict mode is on, calling
    :func:`validate_or_raise` raises :class:`PtxValidationError` on any
    error-severity issue. Warnings (e.g. unknown opcodes) never raise.
  * A context manager :func:`strict` for temporary toggling.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Iterable

from pyptx.ir.nodes import Instruction
from pyptx.spec.ptx import (
    INSTRUCTIONS,
    InstructionSpec,
    ModifierGroup,
    _ALL_TYPES,
    _CACHE,
    _SPACE,
    _TYPE,
    _VEC,
    _WGMMA_DTYPES_AB,
    _WGMMA_DTYPES_D,
)


# ---------------------------------------------------------------------------
# Issues / exceptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation issue found in an instruction."""

    instruction: Instruction
    message: str
    severity: str = "error"  # "error" or "warning"

    def __str__(self) -> str:
        opcode = self.instruction.opcode + "".join(self.instruction.modifiers)
        return f"[{self.severity}] {opcode}: {self.message}"


# Backwards compatibility alias — older code (and test_spec.py) imports
# ``ValidationError`` as the dataclass type. Keep the old name pointing at
# the new dataclass so nothing breaks.
ValidationError = ValidationIssue


class PtxValidationError(Exception):
    """Raised when an instruction fails strict validation.

    Wraps a list of :class:`ValidationIssue` objects with a readable
    aggregate message that names the offending opcode, lists each issue,
    and pinpoints the user's source line if it could be determined.
    """

    def __init__(
        self,
        issues: Iterable[ValidationIssue],
        *,
        user_frame: str | None = None,
    ) -> None:
        self.issues: list[ValidationIssue] = list(issues)
        self.user_frame: str | None = user_frame
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if not self.issues:
            return "PTX validation failed (no issues recorded)"
        first = self.issues[0].instruction
        opcode_chain = first.opcode + "".join(first.modifiers)
        lines = [opcode_chain]
        for issue in self.issues:
            if issue.severity == "error":
                lines.append(f"  {issue.message}")
        for issue in self.issues:
            if issue.severity == "warning":
                lines.append(f"  [warning] {issue.message}")
        if self.user_frame:
            lines.append(f"  at {self.user_frame}")
        return "\n".join(lines)


class UnvalidatedInstructionWarning(UserWarning):
    """Emitted when an opcode has no spec to validate against.

    Silenced by default (use ``warnings.simplefilter('always',
    UnvalidatedInstructionWarning)`` in tests / debug to surface them).
    """


warnings.simplefilter("ignore", UnvalidatedInstructionWarning)


# ---------------------------------------------------------------------------
# Strict mode controls
# ---------------------------------------------------------------------------


_strict_mode: bool = True


def set_strict(enabled: bool) -> None:
    """Enable or disable strict validation globally.

    When strict mode is on (the default), :func:`validate_or_raise` raises
    :class:`PtxValidationError` on any error-severity issue. When off,
    issues are collected but no exception is raised.
    """
    global _strict_mode
    _strict_mode = bool(enabled)


def is_strict() -> bool:
    """Return whether strict validation is currently enabled."""
    return _strict_mode


@contextmanager
def strict(enabled: bool) -> Generator[None, None, None]:
    """Temporarily enable or disable strict validation.

    Usage::

        with strict(False):
            kernel(...)  # validation issues are collected, never raised
    """
    global _strict_mode
    prev = _strict_mode
    _strict_mode = bool(enabled)
    try:
        yield
    finally:
        _strict_mode = prev


# ---------------------------------------------------------------------------
# Overload registry: opcode -> list[InstructionSpec]
# ---------------------------------------------------------------------------
#
# The base table in ``pyptx.spec.ptx`` keys specs by opcode, so an opcode
# like ``cp`` (which has both a normal ``cp.async.bulk.tensor.*`` form and
# a ``cp.reduce.async.bulk.tensor.*`` form) only retains the last
# registration. This registry holds *all* specs for an opcode (including
# the one in the base table) so the validator can pick the best match.

_OVERLOADS: dict[str, list[InstructionSpec]] = {}
_SPEC_CACHE: dict[str, tuple[InstructionSpec, ...]] = {}


def register_overload(spec: InstructionSpec) -> None:
    """Register an additional spec for an opcode.

    Narrow specs are inserted ahead of any broad doc specs already
    present so the validator's early-exit on a zero-error narrow match
    is independent of import order.
    """
    lst = _OVERLOADS.setdefault(spec.opcode, [])
    _SPEC_CACHE.pop(spec.opcode, None)
    if spec.broad:
        lst.append(spec)
        return
    for i, existing in enumerate(lst):
        if existing.broad:
            lst.insert(i, spec)
            return
    lst.append(spec)


def _candidate_specs(opcode: str) -> tuple[InstructionSpec, ...]:
    """Return the cached spec tuple for ``opcode`` (base + overloads)."""
    cached = _SPEC_CACHE.get(opcode)
    if cached is None:
        specs: list[InstructionSpec] = []
        base = INSTRUCTIONS.get(opcode)
        if base is not None:
            specs.append(base)
        specs.extend(_OVERLOADS.get(opcode, ()))
        cached = tuple(specs)
        _SPEC_CACHE[opcode] = cached
    return cached


def get_specs(opcode: str) -> list[InstructionSpec]:
    """Return every spec registered for ``opcode`` (overloads + base)."""
    return list(_candidate_specs(opcode))


# ---------------------------------------------------------------------------
# Additional specs for instructions that get clobbered or are missing
# ---------------------------------------------------------------------------
#
# These are registered as overloads in this module so they coexist with
# the base table without requiring edits to ``pyptx/spec/ptx.py``.

_PRED_TYPE = ModifierGroup("type", (".pred",), required=True)
# .cluster scope is a Hopper+ extension to the base ``.cta/.gpu/.sys`` set.
_SCOPE_HOPPER = ModifierGroup("scope", (".cta", ".cluster", ".gpu", ".sys"))
_SEM = ModifierGroup("sem", (".weak", ".volatile", ".relaxed", ".acquire", ".release", ".sc", ".acq_rel"))


def _seed_overloads() -> None:
    # ---- cp.async.bulk.tensor (TMA) -------------------------------------
    # The base table currently keeps only the cp.reduce variant; register
    # the plain TMA copy as an overload so the typed cp.async.bulk wrappers
    # validate.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("tensor_kind", (".tile", ".im2col", ".im2col_no_offs", ".gather4")),
            ModifierGroup("dst", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("completion", (
                ".mbarrier::complete_tx::bytes",
                ".bulk_group",
            )),
            ModifierGroup("multicast", (".multicast::cluster",)),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[dst], [src], size|tensorCoords, [mbar] [, ctaMask] [, policy]",
        min_operands=2,
        max_operands=10,
        description="Asynchronous bulk copy / TMA tensor load-store (Hopper)",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # Plain cp.async (Ampere-style cp.async.cg/.ca to shared memory).
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("cache", (".ca", ".cg")),
            ModifierGroup("dst", (".shared", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global",)),
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
        ),
        operand_pattern="[dst], [src], byteCount",
        min_operands=2,
        max_operands=4,
        description="cp.async (Ampere) — async copy global → shared",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.bulk.commit_group — closes all preceding
    # cp.async.bulk.*.bulk_group operations into a single commit group.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("action", (".commit_group",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Commit pending cp.async.bulk.*.bulk_group ops",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # cp.async.bulk.wait_group N  (optionally cp.async.bulk.wait_group.read)
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("action", (".wait_group",), required=True),
            ModifierGroup("relax", (".read",)),
        ),
        operand_pattern="N",
        min_operands=1,
        max_operands=1,
        description="Wait until at most N bulk commit groups remain",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # ---- Ampere plain cp.async commit/wait (no .bulk) ---------------------
    # cp.async.commit_group — closes pending cp.async.{cg,ca} into a group.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".commit_group",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="cp.async.commit_group (Ampere) — close pending cp.async into a group",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.wait_group N — wait until at most N groups remain pending.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".wait_group",), required=True),
        ),
        operand_pattern="N",
        min_operands=1,
        max_operands=1,
        description="cp.async.wait_group N (Ampere) — wait until <= N groups pending",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # cp.async.wait_all — wait for all pending cp.async to complete.
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("action", (".wait_all",), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="cp.async.wait_all (Ampere) — wait for all pending cp.async",
        since_version=(7, 0),
        arch="sm_80",
    ))

    # ---- tcgen05 (Blackwell) ---------------------------------------------
    # The base table holds a stripped-down tcgen05 spec; the full family
    # (alloc/dealloc/ld/st/cp/shift/commit/fence/wait/relinquish) is
    # registered here as an overload.
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (
                ".mma", ".ld", ".st", ".cp", ".shift",
                ".fence", ".commit", ".wait",
                ".alloc", ".dealloc", ".relinquish_alloc_permit",
                ".fence::before_thread_sync", ".fence::after_thread_sync",
                ".wait::ld", ".wait::st",
            ), required=True),
            ModifierGroup("sp", (".sp",)),
            ModifierGroup("ws", (".ws",)),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("kind", (
                ".kind::tf32", ".kind::f16", ".kind::i8",
                ".kind::f8f6f4", ".kind::mxf8f6f4",
                ".kind::mxf4", ".kind::mxf4nvf4",
            )),
            ModifierGroup("ashift", (".ashift",)),
            ModifierGroup("collector", (
                ".collector::a::discard",
                ".collector::a::lastuse",
                ".collector::a::fill",
                ".collector::a::use",
            )),
            ModifierGroup("block_scale", (".block_scale",)),
            ModifierGroup("scale_vec_size",
                          (".scale_vec_size::1X", ".scale_vec_size::2X", ".scale_vec_size::4X")),
            ModifierGroup("scale_vec",
                          (".scale_vec::1X", ".scale_vec::2X", ".scale_vec::4X")),
            ModifierGroup("scale_block", (".block16", ".block32")),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", (
                ".16x32bx2", ".16x64b", ".16x128b", ".16x256b",
                ".31x256b", ".32x32b", ".4x256b", ".32x128b",
                ".64x128b", ".128x128b", ".128x256b",
            )),
            ModifierGroup("num", (
                ".x1", ".x2", ".x4", ".x8", ".x16", ".x32", ".x64", ".x128",
            )),
            ModifierGroup("pack", (".pack::16b", ".unpack::16b")),
            ModifierGroup("warp_layout", (".warpx2::02_13", ".warpx2::01_23", ".warpx4")),
            ModifierGroup("subbyte_layout", (".b8x16",)),
            ModifierGroup("subbyte_pack", (".b4x16_p64", ".b6x16_p32", ".b6p2x16")),
            ModifierGroup("direction", (".down",)),
            ModifierGroup("space", (".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b32", ".b64")),
            ModifierGroup("completion", (
                ".mbarrier::arrive::one",
                ".multicast::cluster", ".multicast::cluster::all",
            )),
        ),
        operand_pattern="varies by op (tmem addr, descriptors, mbar, regs)",
        min_operands=0,
        max_operands=128,
        description="Fifth-generation tensor core operations (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # ---- mbarrier (extended Hopper variants) -----------------------------
    # The base spec already covers .init/.arrive/.try_wait/etc.; this
    # overload allows the compound forms used on Hopper such as
    # ``mbarrier.arrive.expect_tx`` and ``mbarrier.try_wait.parity``.
    register_overload(InstructionSpec(
        opcode="mbarrier",
        modifier_groups=(
            ModifierGroup("op", (
                ".init", ".inval",
                ".arrive", ".arrive_drop",
                ".test_wait", ".try_wait",
                ".pending_count",
                ".expect_tx", ".complete_tx",
            ), required=True),
            ModifierGroup("variant", (
                ".expect_tx", ".complete_tx",
                ".noComplete", ".parity",
            )),
            ModifierGroup("sem", (".acquire", ".release", ".relaxed")),
            ModifierGroup("scope", (".cta", ".cluster")),
            ModifierGroup("space",
                          (".shared", ".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b64",)),
        ),
        operand_pattern="varies by op",
        min_operands=0,
        max_operands=4,
        description=(
            "Memory barrier object operations (Hopper compound forms: "
            "arrive.expect_tx, try_wait.parity, ...)"
        ),
        since_version=(7, 8),
        arch="sm_90",
    ))

    # ---- barrier.cluster (Hopper) ---------------------------------------
    # The base ``barrier`` spec covers .sync/.arrive with optional .cta
    # /.cluster scope. This overload models the explicit
    # ``barrier.cluster.arrive`` and ``barrier.cluster.wait`` forms.
    register_overload(InstructionSpec(
        opcode="barrier",
        modifier_groups=(
            ModifierGroup("level", (".cluster",), required=True),
            ModifierGroup("op", (".arrive", ".wait"), required=True),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("release", (".release", ".acquire", ".relaxed")),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=1,
        description="Cluster barrier arrive/wait (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ))

    # ---- bar.warp.sync ---------------------------------------------------
    register_overload(InstructionSpec(
        opcode="bar",
        modifier_groups=(
            ModifierGroup("level", (".warp",), required=True),
            ModifierGroup("op", (".sync",), required=True),
        ),
        operand_pattern="memberMask",
        min_operands=0,
        max_operands=1,
        description="Warp-level barrier sync (Volta+)",
        since_version=(6, 0),
    ))

    # ---- Legacy and Blackwell matrix instructions ------------------------
    _mma_shapes = (
        ".m8n8k4", ".m8n8k16", ".m16n8k4", ".m16n8k8",
        ".m16n8k16", ".m16n8k32", ".m16n8k64", ".m16n8k128",
        ".m16n8k256", ".m16n16k4", ".m16n16k8", ".m16n16k16",
    )
    register_overload(InstructionSpec(
        opcode="mma",
        modifier_groups=(
            ModifierGroup("op", (".sync",), required=True),
            ModifierGroup("sparse", (".sp", ".sp::ordered_metadata")),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", _mma_shapes),
            ModifierGroup("layout_a", (".row", ".col")),
            ModifierGroup("layout_b", (".row", ".col")),
            ModifierGroup("dtype_d", _ALL_TYPES),
            ModifierGroup("dtype_a", _ALL_TYPES),
            ModifierGroup("dtype_b", _ALL_TYPES),
            ModifierGroup("dtype_c", _ALL_TYPES),
            ModifierGroup("kind", (
                ".kind::f16", ".kind::tf32", ".kind::f8f6f4",
                ".kind::mxf8f6f4", ".kind::mxf4", ".kind::mxf4nvf4",
            )),
            ModifierGroup("block_scale", (".block_scale",)),
            ModifierGroup("scale_vec_size", (".scale_vec_size::1X", ".scale_vec_size::2X", ".scale_vec_size::4X")),
        ),
        operand_pattern="d, a, b, c [, metadata] [, scale]",
        min_operands=4,
        max_operands=8,
        description="Warp-level matrix multiply-accumulate",
        since_version=(6, 4),
    ))

    register_overload(InstructionSpec(
        opcode="wmma",
        modifier_groups=(
            ModifierGroup("op", (".load", ".store", ".mma"), required=True),
            ModifierGroup("fragment", (".a", ".b", ".c", ".d")),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("layout_a", (".row", ".col")),
            ModifierGroup("layout_b", (".row", ".col")),
            ModifierGroup("shape", (
                ".m8n8k4", ".m16n16k8", ".m16n16k16", ".m32n8k16", ".m8n32k16",
            )),
            ModifierGroup("space", (".global", ".shared")),
            ModifierGroup("dtype_d", _WGMMA_DTYPES_D),
            ModifierGroup("dtype_a", _WGMMA_DTYPES_AB),
            ModifierGroup("dtype_b", _WGMMA_DTYPES_AB),
            ModifierGroup("dtype_c", _ALL_TYPES),
        ),
        operand_pattern="varies by load/store/mma form",
        min_operands=2,
        max_operands=4,
        description="Warp-level WMMA load/store/mma instruction family",
        since_version=(6, 0),
    ))

    register_overload(InstructionSpec(
        opcode="wgmma",
        modifier_groups=(
            ModifierGroup("op", (".mma_async", ".fence", ".commit_group", ".wait_group"), required=True),
            ModifierGroup("sp", (".sp",)),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("satfinite", (".satfinite",)),
            ModifierGroup("shape", (
                ".m64n8k8", ".m64n16k8", ".m64n24k8", ".m64n32k8",
                ".m64n64k8", ".m64n128k8", ".m64n192k8", ".m64n256k8",
                ".m64n8k16", ".m64n16k16", ".m64n24k16", ".m64n32k16",
                ".m64n64k16", ".m64n128k16", ".m64n192k16", ".m64n256k16",
                ".m64n8k32", ".m64n16k32", ".m64n24k32", ".m64n32k32",
                ".m64n64k32", ".m64n128k32", ".m64n192k32", ".m64n256k32",
                ".m64n8k256", ".m64n16k256", ".m64n24k256", ".m64n32k256",
                ".m64n48k256", ".m64n64k256", ".m64n80k256", ".m64n88k256",
                ".m64n96k256", ".m64n112k256", ".m64n128k256", ".m64n144k256",
                ".m64n160k256", ".m64n176k256", ".m64n192k256", ".m64n208k256",
                ".m64n224k256", ".m64n240k256", ".m64n248k256", ".m64n256k256",
            )),
            ModifierGroup("dtype_d", _WGMMA_DTYPES_D),
            ModifierGroup("dtype_a", _WGMMA_DTYPES_AB),
            ModifierGroup("dtype_b", _WGMMA_DTYPES_AB),
            ModifierGroup("bit_op", (".and", ".xor")),
            ModifierGroup("population", (".popc",)),
        ),
        operand_pattern="varies by wgmma form",
        min_operands=0,
        max_operands=128,
        description="Warpgroup matrix instructions, including bit-matrix popcount forms",
        since_version=(8, 0),
        arch="sm_90a",
    ))

    # ---- Scalar math, predicates, conversion and address-space helpers ----
    register_overload(InstructionSpec(
        opcode="cvt",
        modifier_groups=(
            ModifierGroup("rounding", (".rn", ".rz", ".rm", ".rp", ".rni", ".rzi", ".rmi", ".rpi", ".rs")),
            ModifierGroup("ftz", (".ftz",)),
            ModifierGroup("sat", (".sat", ".satfinite")),
            ModifierGroup("relu", (".relu",)),
            ModifierGroup("dst_type", _ALL_TYPES, required=True),
            ModifierGroup("src_type", _ALL_TYPES, required=True),
        ),
        operand_pattern="d, a[, b]",
        min_operands=2,
        max_operands=3,
        description="Convert between scalar and packed PTX types",
        since_version=(1, 0),
    ))

    register_overload(InstructionSpec(
        opcode="div",
        modifier_groups=(
            ModifierGroup("rounding", (".rn", ".rz", ".rm", ".rp")),
            ModifierGroup("mode", (".approx", ".full")),
            ModifierGroup("ftz", (".ftz",)),
            _TYPE,
        ),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Divide, including floating full/approx forms",
    ))

    register_overload(InstructionSpec(
        opcode="ex2",
        modifier_groups=(
            ModifierGroup("mode", (".approx",)),
            ModifierGroup("ftz", (".ftz",)),
            ModifierGroup("type", (".f32",), required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Base-2 exponential approximation",
    ))

    register_overload(InstructionSpec(
        opcode="bfe",
        modifier_groups=(ModifierGroup("type", (".u32", ".s32", ".u64", ".s64", ".b32", ".b64"), required=True),),
        operand_pattern="d, a, b, c",
        min_operands=4,
        max_operands=4,
        description="Bit-field extract",
    ))
    register_overload(InstructionSpec(
        opcode="bfind",
        modifier_groups=(
            ModifierGroup("mode", (".shiftamt",)),
            ModifierGroup("type", (".u32", ".s32", ".u64", ".s64"), required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Find leading set bit",
    ))
    register_overload(InstructionSpec(
        opcode="bmsk",
        modifier_groups=(
            ModifierGroup("mode", (".clamp", ".wrap")),
            ModifierGroup("type", (".b32", ".b64"), required=True),
        ),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Generate bit mask",
    ))
    register_overload(InstructionSpec(
        opcode="isspacep",
        modifier_groups=(ModifierGroup(
            "space",
            (".global", ".shared", ".shared::cta", ".shared::cluster", ".local", ".const", ".param"),
            required=True,
        ),),
        operand_pattern="p, a",
        min_operands=2,
        max_operands=2,
        description="Test whether an address lies in a state space",
    ))

    for opcode in ("and", "or", "xor"):
        register_overload(InstructionSpec(
            opcode=opcode,
            modifier_groups=(_PRED_TYPE,),
            operand_pattern="d, a, b",
            min_operands=3,
            max_operands=3,
            description=f"Predicate {opcode}",
        ))
    register_overload(InstructionSpec(
        opcode="not",
        modifier_groups=(_PRED_TYPE,),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Predicate not",
    ))

    register_overload(InstructionSpec(
        opcode="setp",
        modifier_groups=(
            ModifierGroup("cmp", (
                ".eq", ".ne", ".lt", ".le", ".gt", ".ge",
                ".lo", ".ls", ".hi", ".hs",
                ".equ", ".neu", ".ltu", ".leu", ".gtu", ".geu",
                ".num", ".nan",
            ), required=True),
            ModifierGroup("ftz", (".ftz",)),
            ModifierGroup("boolop", (".and", ".or", ".xor")),
            _TYPE,
        ),
        operand_pattern="p[|q], a, b[, c]",
        min_operands=3,
        max_operands=4,
        description="Set predicate, optionally combining with an input predicate",
    ))

    register_overload(InstructionSpec(
        opcode="cvta",
        modifier_groups=(
            ModifierGroup("direction", (".to",)),
            _SPACE,
            ModifierGroup("size", (".u32", ".u64"), required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Convert address between generic and explicit state space",
    ))

    register_overload(InstructionSpec(
        opcode="membar",
        modifier_groups=(ModifierGroup("level", (".cta", ".gl", ".sys", ".cluster"), required=True),),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Memory barrier, including cluster scope",
    ))

    # ---- Memory operations ------------------------------------------------
    register_overload(InstructionSpec(
        opcode="st",
        modifier_groups=(
            ModifierGroup("async", (".async",)),
            _SEM,
            _SCOPE_HOPPER,
            _SPACE,
            _CACHE,
            _VEC,
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
            _TYPE,
        ),
        operand_pattern="[a], b[, [mbar]]",
        min_operands=2,
        max_operands=3,
        description="Store, including volatile/release/async variants",
    ))
    register_overload(InstructionSpec(
        opcode="st",
        modifier_groups=(
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("sem", (".weak",), required=True),
            ModifierGroup("space", (".shared::cta",), required=True),
        ),
        operand_pattern="[addr], size, initval",
        min_operands=3,
        max_operands=3,
        description="Bulk shared-memory store initialization",
        since_version=(8, 6),
        arch="sm_100",
    ))

    register_overload(InstructionSpec(
        opcode="atom",
        modifier_groups=(
            _SEM,
            _SCOPE_HOPPER,
            _SPACE,
            ModifierGroup("op", (
                ".add", ".min", ".max", ".inc", ".dec",
                ".and", ".or", ".xor", ".exch", ".cas",
            ), required=True),
            ModifierGroup("ftz", (".noftz",)),
            _TYPE,
        ),
        operand_pattern="d, [a], b[, c]",
        min_operands=3,
        max_operands=4,
        description="Atomic read-modify-write with memory semantics",
    ))

    register_overload(InstructionSpec(
        opcode="red",
        modifier_groups=(
            ModifierGroup("async", (".async",)),
            _SEM,
            _SCOPE_HOPPER,
            _SPACE,
            ModifierGroup("op", (".add", ".min", ".max", ".inc", ".dec", ".and", ".or", ".xor"), required=True),
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
            _TYPE,
        ),
        operand_pattern="[a], b[, [mbar]]",
        min_operands=2,
        max_operands=3,
        description="Reduction, including async memory-semantic forms",
    ))

    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("prefetch", (".prefetch",), required=True),
            ModifierGroup("level", (".L2",), required=True),
            ModifierGroup("space", (".global",), required=True),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[src][, policy]",
        min_operands=1,
        max_operands=3,
        description="cp.async.bulk.prefetch to L2",
        since_version=(8, 0),
        arch="sm_90",
    ))
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("prefetch", (".prefetch",), required=True),
            ModifierGroup("tensor", (".tensor",), required=True),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d"), required=True),
            ModifierGroup("level", (".L2",), required=True),
            ModifierGroup("space", (".global",), required=True),
            ModifierGroup("load_mode", (
                ".tile", ".im2col", ".im2col_no_offs",
                ".tile::gather4", ".tile::scatter4",
                ".im2col::w", ".im2col::w::128",
            )),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[tensorMap, coords][, im2colOffsets][, policy]",
        min_operands=1,
        max_operands=3,
        description="cp.async.bulk.prefetch.tensor to L2",
        since_version=(8, 0),
        arch="sm_90",
    ))
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("reduce", (".reduce",), required=True),
            ModifierGroup("op_async", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("dst", (".global",)),
            ModifierGroup("src", (".shared::cta", ".shared::cluster")),
            ModifierGroup("red_op", (".add", ".min", ".max", ".inc", ".dec", ".and", ".or", ".xor"), required=True),
            ModifierGroup("load_mode", (".tile", ".im2col", ".im2col_no_offs")),
            ModifierGroup("completion", (".bulk_group",)),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[tensorMap, coords], [srcMem][, policy]",
        min_operands=2,
        max_operands=6,
        description="cp.reduce.async.bulk.tensor with cache hints",
        since_version=(8, 0),
        arch="sm_90",
    ))
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("mbarrier", (".mbarrier",), required=True),
            ModifierGroup("action", (".arrive",), required=True),
            ModifierGroup("noinc", (".noinc",)),
        ),
        operand_pattern="[mbar]",
        min_operands=1,
        max_operands=1,
        description="cp.async.mbarrier.arrive",
        since_version=(7, 8),
        arch="sm_80",
    ))

    # ---- Tensor map and cluster launch control ---------------------------
    register_overload(InstructionSpec(
        opcode="tensormap",
        modifier_groups=(
            ModifierGroup("op", (".replace",), required=True),
            ModifierGroup("mode", (".tile",), required=True),
            ModifierGroup("field", (
                ".global_address", ".rank",
                ".box_dim", ".global_dim", ".global_stride", ".element_stride",
                ".elemtype", ".interleave_layout", ".swizzle_mode",
                ".swizzle_atomicity", ".fill_mode",
            ), required=True),
            ModifierGroup("space", (".global", ".shared::cta")),
            ModifierGroup("size", (".b1024",), required=True),
            ModifierGroup("type", (".b32", ".b64"), required=True),
        ),
        operand_pattern="[addr], new_val | [addr], ord, new_val",
        min_operands=2,
        max_operands=3,
        description="Modify a tensor-map object field",
        since_version=(8, 3),
        arch="sm_90a",
    ))
    register_overload(InstructionSpec(
        opcode="tensormap",
        modifier_groups=(
            ModifierGroup("op", (".cp_fenceproxy",), required=True),
            ModifierGroup("dst", (".global",), required=True),
            ModifierGroup("src", (".shared::cta",), required=True),
            ModifierGroup("proxy", (".tensormap::generic",), required=True),
            ModifierGroup("sem", (".release",), required=True),
            _SCOPE_HOPPER,
            ModifierGroup("sync", (".sync",), required=True),
            ModifierGroup("aligned", (".aligned",), required=True),
        ),
        operand_pattern="[dst], [src], size",
        min_operands=3,
        max_operands=3,
        description="Copy a tensor map through the fence proxy",
        since_version=(8, 3),
        arch="sm_90",
    ))

    register_overload(InstructionSpec(
        opcode="clusterlaunchcontrol",
        modifier_groups=(
            ModifierGroup("op", (".try_cancel",), required=True),
            ModifierGroup("async", (".async",), required=True),
            ModifierGroup("space", (".shared::cta",), required=True),
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",), required=True),
            ModifierGroup("multicast", (".multicast::cluster::all",)),
            ModifierGroup("type", (".b128",), required=True),
        ),
        operand_pattern="[response], [mbar]",
        min_operands=2,
        max_operands=2,
        description="Try to cancel a cluster launch",
        since_version=(8, 6),
        arch="sm_100",
    ))
    register_overload(InstructionSpec(
        opcode="clusterlaunchcontrol",
        modifier_groups=(
            ModifierGroup("op", (".query_cancel",), required=True),
            ModifierGroup("field", (
                ".is_canceled",
                ".get_first_ctaid::x",
                ".get_first_ctaid::y",
                ".get_first_ctaid::z",
                ".get_first_ctaid",
            ), required=True),
            ModifierGroup("vector", (".v4",)),
            ModifierGroup("dst_type", (".pred", ".b32"), required=True),
            ModifierGroup("src_type", (".b128",), required=True),
        ),
        operand_pattern="d, response",
        min_operands=2,
        max_operands=2,
        description="Query a cluster launch cancellation response",
        since_version=(8, 6),
        arch="sm_100",
    ))

    # ---- Misc newer/generated instruction families -----------------------
    register_overload(InstructionSpec(
        opcode="multimem",
        modifier_groups=(
            ModifierGroup("op", (".ld_reduce", ".red", ".st"), required=True),
            _SEM,
            _SCOPE_HOPPER,
            ModifierGroup("space", (".global",), required=True),
            ModifierGroup("op_kind", (".add", ".min", ".max", ".inc", ".dec", ".and", ".or", ".xor"),),
            ModifierGroup("acc", (".acc::f16",)),
            _TYPE,
        ),
        operand_pattern="varies by multimem op",
        min_operands=2,
        max_operands=4,
        description="Multimem load-reduce/store/reduction family",
        since_version=(8, 1),
        arch="sm_90",
    ))
    register_overload(InstructionSpec(
        opcode="multimem",
        modifier_groups=(
            ModifierGroup("op", (".cp",), required=True),
            ModifierGroup("reduce", (".reduce",)),
            ModifierGroup("async", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",), required=True),
            ModifierGroup("dst", (".global", ".shared::cta", ".shared::cluster"), required=True),
            ModifierGroup("src", (".global", ".shared::cta", ".shared::cluster"), required=True),
        ),
        operand_pattern="[dst], [src], size[, ...]",
        min_operands=2,
        max_operands=8,
        description="Multimem async bulk copy forms",
        since_version=(8, 0),
        arch="sm_90",
    ))
    register_overload(InstructionSpec(
        opcode="trap",
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Trap execution",
    ))

    # ---- Documentation manifest fallback specs ---------------------------
    # Broad specs cover every documented opcode so unmodeled instructions
    # don't degrade to "unknown opcode" warnings. Narrow overloads above
    # still win when they match.
    _doc_modifiers = (
        *_ALL_TYPES,
        ".cc", ".add", ".min", ".max", ".inc", ".dec", ".and", ".or", ".xor",
        ".cp", ".reduce", ".mbarrier", ".arrive", ".noinc", ".pack", ".ws",
        ".eq", ".ne", ".lt", ".le", ".gt", ".ge", ".lo", ".ls", ".hi", ".hs",
        ".equ", ".neu", ".ltu", ".leu", ".gtu", ".geu", ".num", ".nan",
        ".wide", ".rn", ".rz", ".rm", ".rp", ".rni", ".rzi",
        ".rmi", ".rpi", ".approx", ".full", ".ftz", ".noftz", ".sat",
        ".satfinite", ".relu", ".NaN", ".xorsign.abs", ".abs",
        ".shiftamt", ".wrap", ".clamp", ".zero", ".trap", ".idx", ".uni",
        ".number", ".notanumber", ".normal", ".subnormal", ".infinite",
        ".sync", ".aligned", ".volatile", ".weak", ".relaxed", ".acquire",
        ".release", ".sc", ".acq_rel", ".global", ".local", ".shared",
        ".shared::cta", ".shared::cluster", ".const", ".param",
        ".cta", ".cluster", ".gpu", ".sys", ".ca", ".cg", ".cs", ".lu",
        ".cv", ".nc", ".L1", ".L2", ".L2::cache_hint", ".evict_first",
        ".evict_last", ".evict_normal", ".cache_hint", ".v2", ".v4",
        ".1d", ".2d", ".3d", ".a1d", ".a2d", ".cube", ".acube", ".2dms",
        ".a2dms", ".level", ".grad", ".bias", ".lod", ".lz", ".finite",
        ".array", ".bound", ".normalized", ".geometry", ".samplepos",
        ".channel_data_type", ".channel_order", ".width", ".height",
        ".depth", ".layers", ".samples", ".format", ".mode", ".bulk",
        ".async", ".priority", ".fractional", ".discard", ".b", ".h", ".w",
        ".d", ".q", ".l", ".r", ".g", ".a", ".x", ".y", ".z",
        ".m8n8", ".m8n8k4", ".m8n8k16", ".m8n8k32", ".m8n8k128",
        ".m16n8k4", ".m16n8k8", ".m16n8k16", ".m16n8k32",
        ".m16n8k64", ".m16n8k128", ".m16n8k256", ".m16n16k8",
        ".m16n16k16", ".m32n8k16", ".m8n32k16", ".x1", ".x2", ".x4",
    )

    # All broad doc specs share the same modifier set; reusing one
    # ModifierGroup means _options_set / _spec_known cache one frozenset
    # for them collectively instead of one per opcode.
    doc_modifier_group = ModifierGroup("modifiers", _doc_modifiers)

    def _register_broad_doc_spec(
        opcode: str,
        min_operands: int,
        max_operands: int,
        description: str,
    ) -> None:
        spec = InstructionSpec(
            opcode=opcode,
            modifier_groups=(doc_modifier_group,),
            operand_pattern="varies by documented PTX form",
            min_operands=min_operands,
            max_operands=max_operands,
            description=description,
            broad=True,
        )
        register_overload(spec)

    for opcode, min_ops, max_ops, desc in (
        # Existing base opcodes with documented forms beyond the narrow
        # table entries in pyptx.spec.ptx.
        ("add", 3, 3, "Add, including extended-precision and FP variants"),
        ("sub", 3, 3, "Subtract, including extended-precision and FP variants"),
        ("mul", 3, 3, "Multiply, including integer, FP, and packed variants"),
        ("mad", 4, 5, "Multiply-add, including extended-precision variants"),
        ("abs", 2, 2, "Absolute value for integer/FP/packed types"),
        ("neg", 2, 2, "Negate for integer/FP/packed types"),
        ("min", 3, 3, "Minimum for integer/FP/packed types"),
        ("max", 3, 3, "Maximum for integer/FP/packed types"),
        ("ld", 2, 4, "Load, including memory semantics and cache variants"),
        ("multimem", 2, 8, "Multimem copy, load-reduce, store, and reduction forms"),
        ("shfl", 4, 5, "Shuffle, deprecated and sync forms"),
        ("vote", 2, 3, "Vote, deprecated and sync forms"),
        ("match", 2, 3, "Match sync forms"),
        ("redux", 2, 3, "Redux sync forms"),
        ("elect", 1, 2, "Elect sync forms"),
        ("cvt", 2, 4, "Convert and cvt.pack forms"),
        # Missing documented base opcodes.
        ("mul24", 3, 3, "24-bit multiply"),
        ("mad24", 4, 4, "24-bit multiply-add"),
        ("sad", 4, 4, "Sum of absolute differences"),
        ("popc", 2, 2, "Population count"),
        ("clz", 2, 2, "Count leading zeros"),
        ("fns", 4, 4, "Find n-th set bit"),
        ("brev", 2, 2, "Bit reverse"),
        ("bfi", 5, 5, "Bit-field insert"),
        ("szext", 4, 4, "Sign/zero extend selected bits"),
        ("dp4a", 4, 4, "Four-way 8-bit integer dot product"),
        ("dp2a", 5, 5, "Two-way 16/8-bit integer dot product"),
        ("addc", 3, 4, "Add with carry"),
        ("subc", 3, 4, "Subtract with carry"),
        ("madc", 4, 5, "Multiply-add with carry"),
        ("testp", 2, 2, "Floating point property test"),
        ("copysign", 3, 3, "Copy floating point sign"),
        ("rcp", 2, 2, "Reciprocal"),
        ("sqrt", 2, 2, "Square root"),
        ("rsqrt", 2, 2, "Reciprocal square root"),
        ("sin", 2, 2, "Sine approximation"),
        ("cos", 2, 2, "Cosine approximation"),
        ("lg2", 2, 2, "Base-2 logarithm"),
        ("tanh", 2, 2, "Hyperbolic tangent approximation"),
        ("set", 3, 4, "Set integer value from comparison"),
        ("slct", 4, 4, "Select based on sign"),
        ("cnot", 2, 2, "Predicate/integer complement-not"),
        ("shf", 4, 4, "Funnel shift"),
        ("ldu", 2, 2, "Uniform load"),
        ("prefetch", 1, 2, "Prefetch data"),
        ("prefetchu", 1, 1, "Prefetch uniform data"),
        ("applypriority", 2, 3, "Apply eviction priority"),
        ("discard", 1, 2, "Discard cache line"),
        ("createpolicy", 2, 4, "Create cache eviction policy"),
        ("tex", 2, 8, "Texture load"),
        ("tld4", 2, 8, "Texture gather"),
        ("txq", 2, 4, "Texture query"),
        ("istypep", 2, 2, "Test texture/surface handle type"),
        ("suld", 2, 6, "Surface load"),
        ("sust", 2, 8, "Surface store"),
        ("sured", 2, 8, "Surface reduction"),
        ("suq", 2, 4, "Surface query"),
        ("brx", 2, 2, "Indirect branch"),
        ("activemask", 1, 1, "Active lane mask"),
        ("movmatrix", 2, 2, "Move matrix registers"),
        ("stacksave", 1, 1, "Save stack pointer"),
        ("stackrestore", 1, 1, "Restore stack pointer"),
        ("alloca", 2, 3, "Allocate local stack memory"),
        ("brkpt", 0, 0, "Breakpoint"),
        ("nanosleep", 1, 1, "Suspend execution for a duration"),
        ("pmevent", 1, 1, "Performance monitor event"),
        ("vadd", 4, 4, "Scalar video add"),
        ("vsub", 4, 4, "Scalar video subtract"),
        ("vabsdiff", 4, 4, "Scalar video absolute difference"),
        ("vmin", 3, 4, "Scalar video minimum"),
        ("vmax", 3, 4, "Scalar video maximum"),
        ("vshl", 3, 4, "Scalar video shift left"),
        ("vshr", 3, 4, "Scalar video shift right"),
        ("vmad", 4, 5, "Scalar video multiply-add"),
        ("vset", 3, 4, "Scalar video comparison"),
        ("vadd2", 4, 4, "SIMD video add2"),
        ("vsub2", 4, 4, "SIMD video subtract2"),
        ("vavrg2", 4, 4, "SIMD video average2"),
        ("vabsdiff2", 4, 4, "SIMD video absolute difference2"),
        ("vmin2", 3, 4, "SIMD video minimum2"),
        ("vmax2", 3, 4, "SIMD video maximum2"),
        ("vset2", 3, 4, "SIMD video comparison2"),
        ("vadd4", 4, 4, "SIMD video add4"),
        ("vsub4", 4, 4, "SIMD video subtract4"),
        ("vavrg4", 4, 4, "SIMD video average4"),
        ("vabsdiff4", 4, 4, "SIMD video absolute difference4"),
        ("vmin4", 3, 4, "SIMD video minimum4"),
        ("vmax4", 3, 4, "SIMD video maximum4"),
        ("vset4", 3, 4, "SIMD video comparison4"),
    ):
        _register_broad_doc_spec(opcode, min_ops, max_ops, desc)


_seed_overloads()


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

# Membership tests on ModifierGroup.options sit on the validation hot
# path. Cache a frozenset per group/spec, keyed by id() — every group
# reaching the validator lives in INSTRUCTIONS or _OVERLOADS for the
# process lifetime, so id reuse cannot occur.
_GROUP_OPTIONS: dict[int, frozenset[str]] = {}
_SPEC_KNOWN: dict[int, frozenset[str]] = {}


def _options_set(group: ModifierGroup) -> frozenset[str]:
    s = _GROUP_OPTIONS.get(id(group))
    if s is None:
        s = frozenset(group.options)
        _GROUP_OPTIONS[id(group)] = s
    return s


def _spec_known(spec: InstructionSpec) -> frozenset[str]:
    s = _SPEC_KNOWN.get(id(spec))
    if s is None:
        s = frozenset().union(*(g.options for g in spec.modifier_groups))
        _SPEC_KNOWN[id(spec)] = s
    return s


def _hint_group_for(
    spec: InstructionSpec,
    mod: str,
) -> ModifierGroup | None:
    """Find the group most likely intended for an unrecognized modifier.

    Heuristic: the group whose option with the longest common prefix
    with ``mod`` wins. Falls back to ``None`` if no overlap exists.
    """
    best: tuple[int, ModifierGroup] | None = None
    for group in spec.modifier_groups:
        for option in group.options:
            i = 0
            limit = min(len(option), len(mod))
            while i < limit and option[i] == mod[i]:
                i += 1
            if i >= 2 and (best is None or i > best[0]):
                best = (i, group)
    return best[1] if best is not None else None


def _validate_against(
    inst: Instruction,
    spec: InstructionSpec,
) -> list[ValidationIssue]:
    """Validate an instruction against a single spec."""
    issues: list[ValidationIssue] = []

    # Track which positions in inst.modifiers have already been consumed
    # by an earlier group, so the same modifier value (e.g. ``.bf16``) can
    # legitimately satisfy two adjacent groups (dtype_a and dtype_b).
    consumed: list[bool] = [False] * len(inst.modifiers)

    for group in spec.modifier_groups:
        opts = _options_set(group)
        match_idx: int | None = None
        for idx, mod in enumerate(inst.modifiers):
            if consumed[idx]:
                continue
            if mod in opts:
                match_idx = idx
                break  # consume the first match; same value can recur

        if match_idx is not None:
            consumed[match_idx] = True
        elif group.required:
            issues.append(ValidationIssue(
                instruction=inst,
                message=(
                    f"Missing required modifier from group '{group.name}': "
                    f"expected one of {group.options}"
                ),
            ))

    remaining_mods = [
        mod for mod, used in zip(inst.modifiers, consumed) if not used
    ]

    # Note: leftover modifiers are treated as *errors* below — when the
    # opcode is in the spec, an unknown modifier means either the user
    # made a typo or our spec is incomplete. Either way the typed surface
    # should hear about it. The escape-hatch path in ``ptx._emit`` calls
    # ``validate_instruction`` directly (not ``validate_or_raise``) and
    # therefore does not surface these as exceptions.

    all_known = _spec_known(spec)
    for mod in remaining_mods:
        if mod not in all_known:
            # Find the group whose name best describes what this slot
            # is *supposed* to hold, to help the user understand what
            # the legal values are.
            hint_group = _hint_group_for(spec, mod)
            if hint_group is not None:
                msg = (
                    f"Unrecognized modifier {mod!r}; "
                    f"expected a value from group {hint_group.name!r} "
                    f"(one of {hint_group.options})"
                )
            else:
                msg = f"Unrecognized modifier {mod!r} for opcode {inst.opcode!r}"
            issues.append(ValidationIssue(
                instruction=inst,
                message=msg,
                severity="error",
            ))

    n_operands = len(inst.operands)
    if n_operands < spec.min_operands:
        issues.append(ValidationIssue(
            instruction=inst,
            message=(
                f"Too few operands: got {n_operands}, "
                f"expected at least {spec.min_operands}"
            ),
        ))
    if n_operands > spec.max_operands:
        issues.append(ValidationIssue(
            instruction=inst,
            message=(
                f"Too many operands: got {n_operands}, "
                f"expected at most {spec.max_operands}"
            ),
        ))

    return issues


def _error_count(issues: Iterable[ValidationIssue]) -> int:
    return sum(1 for i in issues if i.severity == "error")


def _has_missing_required_modifier(issues: Iterable[ValidationIssue]) -> bool:
    return any(i.message.startswith("Missing required modifier") for i in issues)


def validate_instruction(
    inst: Instruction,
    spec_table: dict[str, InstructionSpec] | None = None,
) -> list[ValidationIssue]:
    """Validate an Instruction node against the ISA spec.

    Returns a list of :class:`ValidationIssue`s (empty if the instruction
    is valid). When multiple specs are registered for the opcode, the
    spec yielding the fewest error-severity issues is used.

    If ``spec_table`` is supplied, it overrides the global base table but
    overload entries from this module are still consulted.
    """
    if spec_table is None:
        candidate_specs: tuple[InstructionSpec, ...] | list[InstructionSpec]
        candidate_specs = _candidate_specs(inst.opcode)
    else:
        candidate_specs = []
        base = spec_table.get(inst.opcode)
        if base is not None:
            candidate_specs.append(base)
        candidate_specs.extend(_OVERLOADS.get(inst.opcode, ()))

    if not candidate_specs:
        return [ValidationIssue(
            instruction=inst,
            message=f"Unknown opcode: {inst.opcode!r}",
            severity="warning",
        )]

    best: list[ValidationIssue] | None = None
    best_errors = None
    best_spec: InstructionSpec | None = None
    best_narrow: list[ValidationIssue] | None = None
    best_narrow_errors = None
    for spec in candidate_specs:
        issues = _validate_against(inst, spec)
        n_err = _error_count(issues)
        if not spec.broad:
            if best_narrow is None or n_err < best_narrow_errors:
                best_narrow = issues
                best_narrow_errors = n_err
        if best is None or n_err < best_errors:
            best = issues
            best_errors = n_err
            best_spec = spec
            if n_err == 0 and not spec.broad:
                break

    if (
        best_spec is not None
        and best_spec.broad
        and best_narrow is not None
        and _has_missing_required_modifier(best_narrow)
    ):
        return best_narrow

    return best or []


# ---------------------------------------------------------------------------
# Strict-mode entry point used by ptx._emit
# ---------------------------------------------------------------------------


def _find_user_frame() -> str | None:
    """Walk up the call stack and find the first frame outside of pyptx.

    Returns a string like ``"file.py:42 in fn_name()"`` or ``None`` if
    nothing useful was found.
    """
    import os
    import sys

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    ).replace("\\", "/")
    pkg_root = f"{repo_root}/pyptx"
    preferred: str | None = None
    fallback: str | None = None
    test_frame: str | None = None

    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_code.co_filename
        norm = os.path.abspath(filename).replace("\\", "/")
        # Skip frames inside pyptx itself.
        if "/pyptx/pyptx/" in norm or norm.endswith("/pyptx/spec/validate.py"):
            frame = frame.f_back
            continue
        if "/pyptx/" in norm and "/site-packages/" not in norm:
            # In-tree pyptx files: still skip
            if any(part in norm for part in (
                "/pyptx/ptx.py", "/pyptx/_trace.py", "/pyptx/kernel.py",
                "/pyptx/reg.py", "/pyptx/smem.py", "/pyptx/codegen/",
                "/pyptx/ir/", "/pyptx/parser/", "/pyptx/emitter/",
                "/pyptx/tracer/", "/pyptx/spec/",
            )):
                frame = frame.f_back
                continue
        base = os.path.basename(filename)
        lineno = frame.f_lineno
        funcname = frame.f_code.co_name
        rendered = f"line {lineno} in {funcname}() ({base}:{lineno})"

        # Strongest signal: an actual test file or test function frame.
        if (
            "/tests/" in norm
            or base.startswith("test_")
            or funcname.startswith("test_")
        ):
            if test_frame is None:
                test_frame = rendered
            frame = frame.f_back
            continue

        # Prefer non-library files inside the current repo, such as tests,
        # scripts, or user kernels authored alongside the package.
        if (
            norm.startswith(repo_root + "/")
            and not norm.startswith(pkg_root + "/")
            and "/site-packages/" not in norm
            and "/.venv/" not in norm
        ):
            if preferred is None:
                preferred = rendered
            frame = frame.f_back
            continue

        # Skip common pytest/pluggy wrappers so we can keep walking toward
        # the real user frame instead of stopping at pytest_pyfunc_call().
        if any(part in norm for part in (
            "/site-packages/_pytest/",
            "/site-packages/pluggy/",
            "/site-packages/pytest/",
        )) or base == "pytest":
            if fallback is None:
                fallback = rendered
            frame = frame.f_back
            continue

        if preferred is None:
            preferred = rendered
        frame = frame.f_back
    return test_frame or preferred or fallback


def validate_or_raise(inst: Instruction) -> list[ValidationIssue]:
    """Validate ``inst`` and, in strict mode, raise on errors.

    Always returns the full list of issues. Warnings (e.g. unknown
    opcodes) are surfaced via :class:`UnvalidatedInstructionWarning` and
    never cause an exception.
    """
    issues = validate_instruction(inst)

    # Surface unknown-opcode warnings as Python warnings (suppressed by
    # default — users opt in by adjusting the warning filter).
    has_unknown_opcode = any(
        i.severity == "warning" and i.message.startswith("Unknown opcode")
        for i in issues
    )
    if has_unknown_opcode:
        opcode_chain = inst.opcode + "".join(inst.modifiers)
        warnings.warn(
            f"No spec registered for {opcode_chain!r}; skipping validation",
            UnvalidatedInstructionWarning,
            stacklevel=3,
        )
        return issues

    error_issues = [i for i in issues if i.severity == "error"]
    if error_issues and _strict_mode:
        raise PtxValidationError(
            issues, user_frame=_find_user_frame()
        )

    return issues
