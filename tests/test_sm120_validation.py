"""Coverage for sm_120/Blackwell-era PTX validation."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from pyptx import reg, smem
from pyptx._trace import trace_scope
from pyptx.ir.nodes import Block, Function, ImmediateOperand, Instruction, RegisterOperand
from pyptx.ir.types import ScalarType
from pyptx.parser import parse
from pyptx.spec import validate_instruction
from pyptx.types import b1, e2m1, from_name, s16x2, s8x4, u16x2, u8x4

from tests.conftest import case_id, validation_errors


SM120A_PTXAS_SMOKE = """.version 8.7
.target sm_120a
.address_size 64

.visible .entry smoke(.param .u64 outp) {
  .reg .pred %p<8>;
  .reg .b32 %r<64>;
  .reg .b64 %rd<16>;
  .reg .f32 %f<32>;

  ld.param.u64 %rd0, [outp];
  mov.u32 %r1, 1;
  mov.u32 %r2, 2;
  mov.u32 %r3, 3;
  mov.u32 %r4, 4;
  add.cc.u32 %r5, %r1, %r2;
  addc.u32 %r6, %r3, %r4;
  sub.cc.u32 %r7, %r4, %r1;
  subc.u32 %r8, %r4, %r2;
  mad.lo.cc.u32 %r9, %r1, %r2, %r3;
  madc.lo.u32 %r10, %r1, %r2, %r3;
  popc.b32 %r11, %r10;
  clz.b32 %r12, %r10;
  brev.b32 %r13, %r10;
  bfe.u32 %r14, %r10, 0, 8;
  bfi.b32 %r15, %r1, %r2, 0, 8;
  bfind.u32 %r16, %r10;
  bmsk.clamp.b32 %r17, 4, 8;
  sad.u32 %r18, %r1, %r2, %r3;
  cnot.b32 %r19, %r18;
  shf.l.wrap.b32 %r20, %r1, %r2, 3;
  set.lt.u32.u32 %r21, %r1, %r2;
  setp.lt.u32 %p1, %r1, %r2;
  slct.u32.s32 %r22, %r1, %r2, %r3;
  testp.number.f32 %p2, %f1;
  abs.f32 %f2, %f1;
  neg.f32 %f3, %f2;
  copysign.f32 %f4, %f2, %f3;
  rcp.approx.ftz.f32 %f5, %f4;
  sqrt.rn.ftz.f32 %f6, %f5;
  rsqrt.approx.ftz.f32 %f7, %f6;
  sin.approx.ftz.f32 %f8, %f7;
  cos.approx.ftz.f32 %f9, %f8;
  lg2.approx.ftz.f32 %f10, %f9;
  ex2.approx.ftz.f32 %f11, %f10;
  tanh.approx.f32 %f12, %f11;
  activemask.b32 %r23;
  nanosleep.u32 %r1;
  pmevent 0;
  st.global.u32 [%rd0], %r23;
  ret;
}
"""


def _walk(stmts):
    for stmt in stmts:
        if isinstance(stmt, Instruction):
            yield stmt
        elif isinstance(stmt, Block):
            yield from _walk(stmt.body)


_SCALAR_PROPERTY_AND_MASK_CASES = [
    Instruction(
        opcode="bmsk",
        modifiers=(".clamp", ".b32"),
        operands=(RegisterOperand("%r17"), ImmediateOperand("4"), ImmediateOperand("8")),
    ),
    Instruction(
        opcode="bmsk",
        modifiers=(".wrap", ".b32"),
        operands=(RegisterOperand("%r17"), ImmediateOperand("4"), ImmediateOperand("8")),
    ),
    Instruction(
        opcode="testp",
        modifiers=(".number", ".f32"),
        operands=(RegisterOperand("%p2"), RegisterOperand("%f1")),
    ),
    Instruction(
        opcode="testp",
        modifiers=(".notanumber", ".f32"),
        operands=(RegisterOperand("%p2"), RegisterOperand("%f1")),
    ),
]


@pytest.mark.parametrize("inst", _SCALAR_PROPERTY_AND_MASK_CASES, ids=case_id)
def test_blackwell_scalar_property_and_mask_forms_validate(inst):
    assert validation_errors(inst) == []


def test_ptx_corpus_validates_without_unknown_opcodes_or_errors():
    unknown = []
    errors = []

    for path in sorted(Path("tests/corpus").rglob("*.ptx")):
        module = parse(path.read_text())
        for directive in module.directives:
            if not isinstance(directive, Function):
                continue
            for inst in _walk(directive.body):
                if inst.opcode.startswith(".") or inst.opcode in ("{", "}"):
                    continue
                issues = validate_instruction(inst)
                if any(i.severity == "warning" and i.message.startswith("Unknown opcode") for i in issues):
                    unknown.append((path, inst.opcode, inst.modifiers))
                errors.extend((path, issue) for issue in issues if issue.severity == "error")

    assert unknown == []
    assert errors == []


_MATRIX_AND_TENSOR_MAP_CASES = [
    Instruction(
        opcode="mma",
        modifiers=(".sync", ".aligned", ".m16n8k16", ".row", ".col", ".f32", ".f16", ".f16", ".f32"),
        operands=(RegisterOperand("%d"), RegisterOperand("%a"), RegisterOperand("%b"), RegisterOperand("%c")),
    ),
    Instruction(
        opcode="wmma",
        modifiers=(".mma", ".sync", ".aligned", ".row", ".col", ".m16n16k8", ".f32", ".tf32", ".tf32", ".f32"),
        operands=(RegisterOperand("%d"), RegisterOperand("%a"), RegisterOperand("%b"), RegisterOperand("%c")),
    ),
    Instruction(
        opcode="cp",
        modifiers=(".async", ".bulk", ".prefetch", ".tensor", ".3d", ".L2", ".global", ".tile", ".L2::cache_hint"),
        operands=(RegisterOperand("%rd0"), RegisterOperand("%rd1")),
    ),
    Instruction(
        opcode="tensormap",
        modifiers=(".replace", ".tile", ".global_address", ".shared::cta", ".b1024", ".b64"),
        operands=(RegisterOperand("%rd0"), RegisterOperand("%rd1")),
    ),
    Instruction(
        opcode="tensormap",
        modifiers=(".cp_fenceproxy", ".global", ".shared::cta", ".tensormap::generic", ".release", ".gpu", ".sync", ".aligned"),
        operands=(RegisterOperand("%rd0"), RegisterOperand("%rd1"), ImmediateOperand("128")),
    ),
]


@pytest.mark.parametrize("inst", _MATRIX_AND_TENSOR_MAP_CASES, ids=case_id)
def test_sm120_matrix_and_tensor_map_forms_validate(inst):
    assert validation_errors(inst) == []


_CONTROL_AND_TCGEN05_CASES = [
    Instruction(
        opcode="clusterlaunchcontrol",
        modifiers=(".query_cancel", ".get_first_ctaid::x", ".b32", ".b128"),
        operands=(RegisterOperand("%r0"), RegisterOperand("%rq0")),
    ),
    Instruction(
        opcode="clusterlaunchcontrol",
        modifiers=(".try_cancel", ".async", ".shared::cta", ".mbarrier::complete_tx::bytes", ".b128"),
        operands=(RegisterOperand("%rd0"), RegisterOperand("%rd1")),
    ),
    Instruction(
        opcode="tcgen05",
        modifiers=(".cp", ".cta_group::1", ".128x128b", ".b8x16", ".b4x16_p64"),
        operands=(RegisterOperand("%r0"), RegisterOperand("%rd0")),
    ),
    Instruction(
        opcode="wgmma",
        modifiers=(".mma_async", ".sync", ".aligned", ".m64n256k256", ".s32", ".b1", ".b1", ".and", ".popc"),
        operands=(RegisterOperand("%d"), RegisterOperand("%a"), RegisterOperand("%b"), RegisterOperand("%c")),
    ),
]


@pytest.mark.parametrize("inst", _CONTROL_AND_TCGEN05_CASES, ids=case_id)
def test_sm120_control_and_tcgen05_split_modifier_forms_validate(inst):
    assert validation_errors(inst) == []


def test_blackwell_packed_types_are_public_and_parseable():
    for name in ("e2m1", "e2m3", "e3m2", "e2m1x2", "e2m3x2", "e3m2x2", "ue8m0x2", "s2f6x2"):
        assert from_name(name).ptx == f".{name}"
        assert ScalarType.from_ptx(f".{name}").value == name


def test_sub_byte_shared_memory_alloc_uses_physical_storage_bytes():
    with trace_scope() as ctx:
        first = smem.alloc(e2m1, 10, align=1)
        second = smem.alloc(e2m1, 3, align=1)

    assert ctx.var_decls[0].array_size == 10
    assert ctx.var_decls[1].array_size == 3
    assert first.byte_offset == 0
    assert second.byte_offset == 10


def test_b1_shared_memory_alloc_and_wgmma_layout_use_packed_bits():
    with trace_scope() as ctx:
        raw = smem.alloc(b1, (64, 256), align=1)
        tile = smem.wgmma_tile(b1, (64, 256), major="K", align=1)

    assert ctx.var_decls[0].array_size == 64 * 256 // 8
    assert ctx.var_decls[1].array_size == 64 * 256 // 8
    assert raw.byte_offset == 0
    assert tile.byte_offset == 64 * 256 // 8
    assert tile.gmma_layout.smem_swizzle == "32B"
    assert tile.gmma_layout.stride_byte_offset == 8 * 32


def test_packed_instruction_types_declare_bit_storage_registers():
    with trace_scope() as ctx:
        reg.scalar(u16x2)
        reg.array(u8x4, 2)
        reg.alloc(s16x2)
        reg.alloc_array(s8x4, 2)

    assert [decl.type.ptx for decl in ctx.reg_decls] == [".b32", ".b32", ".b32", ".b32"]


def test_storage_backed_register_initializers_use_storage_mov_modifiers():
    with trace_scope() as ctx:
        reg.scalar(u16x2, init=0)
        reg.from_("smem", u16x2)

    moves = [stmt for stmt in ctx.statements if isinstance(stmt, Instruction)]
    assert [inst.modifiers for inst in moves] == [(".b32",), (".b32",)]


def test_8_bit_storage_backed_register_initializers_are_rejected():
    with trace_scope():
        with pytest.raises(TypeError, match="8-bit register storage"):
            reg.scalar(e2m1, init=0)


def test_b1_is_rejected_for_generic_mov_and_bitwise_validation():
    assert validation_errors(Instruction(
        opcode="mov",
        modifiers=(".b1",),
        operands=(RegisterOperand("%r0"), RegisterOperand("%r1")),
    ))
    assert validation_errors(Instruction(
        opcode="and",
        modifiers=(".b1",),
        operands=(RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2")),
    ))


def test_sm120a_tma_corpus_file_assembles_with_ptxas_when_available():
    ptxas = shutil.which("ptxas")
    if ptxas is None:
        pytest.skip("ptxas is not installed")

    source = Path("tests/corpus/external/triton/matmul_tma_sm120a.ptx")
    if not source.exists():
        pytest.skip(f"{source} is not present")

    with tempfile.TemporaryDirectory(prefix="pyptx_sm120a_ptxas_") as tmpdir:
        output = Path(tmpdir) / "matmul_tma_sm120a.cubin"
        result = subprocess.run(
            [ptxas, "-arch=sm_120a", "-o", str(output), str(source)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert output.stat().st_size > 0


def test_sm120a_representative_scalar_ptx_assembles_with_ptxas_when_available():
    ptxas = shutil.which("ptxas")
    if ptxas is None:
        pytest.skip("ptxas is not installed")

    with tempfile.TemporaryDirectory(prefix="pyptx_sm120a_scalar_ptxas_") as tmpdir:
        source = Path(tmpdir) / "sm120a_scalar_smoke.ptx"
        output = Path(tmpdir) / "sm120a_scalar_smoke.cubin"
        source.write_text(SM120A_PTXAS_SMOKE)
        result = subprocess.run(
            [ptxas, "-arch=sm_120a", "-o", str(output), str(source)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert output.stat().st_size > 0
