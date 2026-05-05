"""Enumerations for the PTX type system, state spaces, and linking directives."""

from __future__ import annotations

from enum import Enum


class ScalarType(Enum):
    """PTX scalar types.

    Values are the PTX text representation (without the leading dot).
    """

    B1 = "b1"
    B8 = "b8"
    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    B128 = "b128"
    U8 = "u8"
    U16 = "u16"
    U16X2 = "u16x2"
    U8X4 = "u8x4"
    U32 = "u32"
    U64 = "u64"
    S8 = "s8"
    S16 = "s16"
    S16X2 = "s16x2"
    S8X4 = "s8x4"
    S32 = "s32"
    S64 = "s64"
    F16 = "f16"
    F16X2 = "f16x2"
    BF16 = "bf16"
    BF16X2 = "bf16x2"
    TF32 = "tf32"
    F32 = "f32"
    F32X2 = "f32x2"
    F64 = "f64"
    E2M1 = "e2m1"
    E2M3 = "e2m3"
    E3M2 = "e3m2"
    E4M3 = "e4m3"
    E5M2 = "e5m2"
    UE8M0 = "ue8m0"
    E2M1X2 = "e2m1x2"
    E2M3X2 = "e2m3x2"
    E3M2X2 = "e3m2x2"
    E4M3X2 = "e4m3x2"
    E5M2X2 = "e5m2x2"
    UE8M0X2 = "ue8m0x2"
    E2M1X4 = "e2m1x4"
    E2M3X4 = "e2m3x4"
    E3M2X4 = "e3m2x4"
    E4M3X4 = "e4m3x4"
    E5M2X4 = "e5m2x4"
    S2F6X2 = "s2f6x2"
    PRED = "pred"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.b32'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> ScalarType:
        """Parse from PTX text (with or without leading dot).

        Raises ValueError if not a known type.
        """
        raw = text.lstrip(".")
        return cls(raw)


class StateSpace(Enum):
    """PTX state spaces."""

    REG = "reg"
    SREG = "sreg"
    CONST = "const"
    GLOBAL = "global"
    LOCAL = "local"
    PARAM = "param"
    SHARED = "shared"
    SHARED_CTA = "shared::cta"
    SHARED_CLUSTER = "shared::cluster"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.shared::cta'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> StateSpace:
        """Parse from PTX text (with or without leading dot)."""
        raw = text.lstrip(".")
        return cls(raw)


class LinkingDirective(Enum):
    """PTX linking directives."""

    VISIBLE = "visible"
    EXTERN = "extern"
    WEAK = "weak"
    COMMON = "common"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.visible'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> LinkingDirective:
        """Parse from PTX text (with or without leading dot)."""
        raw = text.lstrip(".")
        return cls(raw)
