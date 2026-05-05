"""PTX scalar type descriptors.

The public API of this module is the set of singleton :class:`PtxType`
instances such as ``u32``, ``bf16``, ``f32``, and ``pred``.

These type objects are used throughout the DSL:

```python
from pyptx.types import bf16, f32, u32, pred

acc = reg.array(f32, 64)
sA = smem.alloc(bf16, (STAGES, BM, BK))
tid = reg.from_(ptx.special.tid.x(), u32)
p = reg.scalar(pred)
```

The type singletons are intentionally lightweight. They mostly serve as
an explicit bridge between Python code and PTX type spelling.
"""

from __future__ import annotations


class PtxType:
    """A PTX scalar type.

    Singleton instances (bf16, f32, etc.) are the public API.
    """

    __slots__ = (
        "name",
        "bits",
        "storage_name",
        "storage_bits",
        "storage_bytes",
        "memory_bits",
    )

    def __init__(
        self,
        name: str,
        bits: int,
        *,
        storage_name: str | None = None,
        memory_bits: int | None = None,
    ) -> None:
        self.name = name
        self.bits = bits
        self.storage_name = storage_name or name
        if storage_name and storage_name.startswith("b") and storage_name[1:].isdigit():
            self.storage_bits = int(storage_name[1:])
        else:
            self.storage_bits = bits
        self.storage_bytes = max((self.storage_bits + 7) // 8, 1)
        self.memory_bits = memory_bits if memory_bits is not None else self.storage_bits

    @property
    def ptx(self) -> str:
        """PTX text form with leading dot: '.f32'."""
        return f".{self.name}"

    @property
    def storage_ptx(self) -> str:
        """PTX declaration storage type for instruction-only formats."""
        return f".{self.storage_name}"

    @property
    def memory_bytes(self) -> int:
        """Byte size for one byte-addressed element container."""
        return max((self.memory_bits + 7) // 8, 1)

    def memory_size_bytes(self, elements: int) -> int:
        """Packed byte count for ``elements`` memory elements."""
        if elements < 0:
            raise ValueError(f"element count must be non-negative, got {elements}")
        return (elements * self.memory_bits + 7) // 8

    @property
    def mov_ptx(self) -> str:
        """PTX modifier to use for raw register moves of this type."""
        if self.storage_name != self.name:
            if self.storage_bits in (16, 32, 64, 128):
                return f".b{self.storage_bits}"
            raise TypeError(
                f"reg moves for {self.name} require {self.storage_ptx} storage, "
                "but PTX has no legal mov modifier for 8-bit register storage"
            )
        return self.ptx

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PtxType):
            return self.name == other.name
        return NotImplemented


# -- Bit types ---------------------------------------------------------------
b1 = PtxType("b1", 1, storage_name="b32", memory_bits=1)
b8 = PtxType("b8", 8)
b16 = PtxType("b16", 16)
b32 = PtxType("b32", 32)
b64 = PtxType("b64", 64)
b128 = PtxType("b128", 128)

# -- Unsigned integers -------------------------------------------------------
u8 = PtxType("u8", 8)
u16 = PtxType("u16", 16)
u16x2 = PtxType("u16x2", 32, storage_name="b32")
u8x4 = PtxType("u8x4", 32, storage_name="b32")
u32 = PtxType("u32", 32)
u64 = PtxType("u64", 64)

# -- Signed integers ---------------------------------------------------------
s8 = PtxType("s8", 8)
s16 = PtxType("s16", 16)
s16x2 = PtxType("s16x2", 32, storage_name="b32")
s8x4 = PtxType("s8x4", 32, storage_name="b32")
s32 = PtxType("s32", 32)
s64 = PtxType("s64", 64)

# -- Floating point ----------------------------------------------------------
f16 = PtxType("f16", 16)
f16x2 = PtxType("f16x2", 32)
bf16 = PtxType("bf16", 16)
bf16x2 = PtxType("bf16x2", 32)
tf32 = PtxType("tf32", 32)
f32 = PtxType("f32", 32)
f32x2 = PtxType("f32x2", 64, storage_name="b64")
f64 = PtxType("f64", 64)

# -- Alternate FP formats (Hopper/Blackwell) ---------------------------------
e2m1 = PtxType("e2m1", 4, storage_name="b8")
e2m3 = PtxType("e2m3", 6, storage_name="b8")
e3m2 = PtxType("e3m2", 6, storage_name="b8")
e4m3 = PtxType("e4m3", 8, storage_name="b8")
e5m2 = PtxType("e5m2", 8, storage_name="b8")
ue8m0 = PtxType("ue8m0", 8, storage_name="b8")
e2m1x2 = PtxType("e2m1x2", 8, storage_name="b8")
e2m3x2 = PtxType("e2m3x2", 12, storage_name="b16")
e3m2x2 = PtxType("e3m2x2", 12, storage_name="b16")
e4m3x2 = PtxType("e4m3x2", 16, storage_name="b16")
e5m2x2 = PtxType("e5m2x2", 16, storage_name="b16")
ue8m0x2 = PtxType("ue8m0x2", 16, storage_name="b16")
e2m1x4 = PtxType("e2m1x4", 16, storage_name="b16")
e2m3x4 = PtxType("e2m3x4", 24, storage_name="b32")
e3m2x4 = PtxType("e3m2x4", 24, storage_name="b32")
e4m3x4 = PtxType("e4m3x4", 32, storage_name="b32")
e5m2x4 = PtxType("e5m2x4", 32, storage_name="b32")
s2f6x2 = PtxType("s2f6x2", 16, storage_name="b16")

# -- Predicate --------------------------------------------------------------
pred = PtxType("pred", 1)

# -- Lookup by name ----------------------------------------------------------
_BY_NAME: dict[str, PtxType] = {t.name: t for t in [
    b1, b8, b16, b32, b64, b128,
    u8, u16, u16x2, u8x4, u32, u64,
    s8, s16, s16x2, s8x4, s32, s64,
    f16, f16x2, bf16, bf16x2, tf32, f32, f32x2, f64,
    e2m1, e2m3, e3m2, e4m3, e5m2, ue8m0,
    e2m1x2, e2m3x2, e3m2x2, e4m3x2, e5m2x2, ue8m0x2,
    e2m1x4, e2m3x4, e3m2x4, e4m3x4, e5m2x4, s2f6x2,
    pred,
]}


def from_name(name: str) -> PtxType:
    """Look up a PtxType by name (with or without leading dot)."""
    raw = name.lstrip(".")
    t = _BY_NAME.get(raw)
    if t is None:
        raise ValueError(f"Unknown PTX type: {name!r}")
    return t
