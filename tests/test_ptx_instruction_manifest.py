"""Documented PTX ISA opcode coverage.

The manifest below is the base-opcode projection of the NVIDIA PTX ISA
instruction table, excluding pseudo-syntax entries such as predicate guards
and brace operands. It intentionally checks validator visibility for every
documented opcode family, not exact semantic validation for every modifier
combination.
"""

from __future__ import annotations

import pytest

from pyptx.ir.nodes import ImmediateOperand, Instruction, RegisterOperand
from pyptx.spec import get_specs

from tests.conftest import case_id, validation_errors


DOCUMENTED_PTX_BASE_OPCODES = (
    "abs",
    "activemask",
    "add",
    "addc",
    "alloca",
    "and",
    "applypriority",
    "atom",
    "bar",
    "barrier",
    "bfe",
    "bfi",
    "bfind",
    "bmsk",
    "bra",
    "brev",
    "brkpt",
    "brx",
    "call",
    "clz",
    "clusterlaunchcontrol",
    "cnot",
    "copysign",
    "cos",
    "cp",
    "createpolicy",
    "cvt",
    "cvta",
    "discard",
    "div",
    "dp2a",
    "dp4a",
    "elect",
    "ex2",
    "exit",
    "fence",
    "fma",
    "fns",
    "getctarank",
    "griddepcontrol",
    "isspacep",
    "istypep",
    "ld",
    "ldmatrix",
    "ldu",
    "lg2",
    "lop3",
    "mad",
    "mad24",
    "madc",
    "mapa",
    "match",
    "max",
    "mbarrier",
    "membar",
    "min",
    "mma",
    "mov",
    "movmatrix",
    "mul",
    "mul24",
    "multimem",
    "nanosleep",
    "neg",
    "not",
    "or",
    "pmevent",
    "popc",
    "prefetch",
    "prefetchu",
    "prmt",
    "rcp",
    "red",
    "redux",
    "rem",
    "ret",
    "rsqrt",
    "sad",
    "selp",
    "set",
    "setmaxnreg",
    "setp",
    "shf",
    "shfl",
    "shl",
    "shr",
    "sin",
    "slct",
    "sqrt",
    "st",
    "stackrestore",
    "stacksave",
    "stmatrix",
    "sub",
    "subc",
    "suld",
    "suq",
    "sured",
    "sust",
    "szext",
    "tanh",
    "tcgen05",
    "tensormap",
    "tex",
    "testp",
    "tld4",
    "trap",
    "txq",
    "vabsdiff",
    "vabsdiff2",
    "vabsdiff4",
    "vadd",
    "vadd2",
    "vadd4",
    "vavrg2",
    "vavrg4",
    "vmax",
    "vmax2",
    "vmax4",
    "vmin",
    "vmin2",
    "vmin4",
    "vmad",
    "vote",
    "vset",
    "vset2",
    "vset4",
    "vshl",
    "vshr",
    "vsub",
    "vsub2",
    "vsub4",
    "wgmma",
    "wmma",
    "xor",
)


def test_every_documented_ptx_base_opcode_has_a_validator_spec():
    missing = [opcode for opcode in DOCUMENTED_PTX_BASE_OPCODES if not get_specs(opcode)]

    assert missing == []


_REPRESENTATIVE_FALLBACK_CASES = [
    Instruction("add", (".cc", ".u32"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"))),
    Instruction("addc", (".cc", ".u32"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"))),
    Instruction("popc", (".b32",), (RegisterOperand("%r0"), RegisterOperand("%r1"))),
    Instruction("shf", (".l", ".wrap", ".b32"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"), ImmediateOperand("3"))),
    Instruction("set", (".lt", ".u32", ".s32"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"))),
    Instruction("cvt", (".pack", ".sat", ".u8", ".s32", ".b32"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"), RegisterOperand("%r3"))),
    Instruction("cp", (".async", ".mbarrier", ".arrive"), (RegisterOperand("%rd0"),)),
    Instruction("multimem", (".cp", ".async", ".bulk", ".global", ".shared::cta"), (RegisterOperand("%rd0"), RegisterOperand("%rd1"), ImmediateOperand("128"))),
    Instruction("tex", (".2d", ".v4", ".f32", ".s32"), (RegisterOperand("%v0"), RegisterOperand("%tex"), RegisterOperand("%coords"))),
    Instruction("suld", (".2d", ".v4", ".b32", ".trap"), (RegisterOperand("%v0"), RegisterOperand("%surf"))),
    Instruction("stacksave", (".u64",), (RegisterOperand("%rd0"),)),
    Instruction("stackrestore", (".u64",), (RegisterOperand("%rd0"),)),
    Instruction("alloca", (".u64",), (RegisterOperand("%rd0"), ImmediateOperand("64"))),
    Instruction("vadd4", (".u32", ".u32", ".u32", ".sat"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"), RegisterOperand("%r3"))),
    Instruction("movmatrix", (".sync", ".aligned", ".m8n8", ".x1", ".b16"), (RegisterOperand("%r0"), RegisterOperand("%r1"))),
    Instruction("tcgen05", (".mma", ".ws", ".cta_group::1", ".kind::f16"), (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"), RegisterOperand("%r3"), RegisterOperand("%p0"))),
    Instruction("brx", (".idx",), (RegisterOperand("%r0"), RegisterOperand("targets"))),
    Instruction("activemask", (".b32",), (RegisterOperand("%r0"),)),
]


@pytest.mark.parametrize("inst", _REPRESENTATIVE_FALLBACK_CASES, ids=case_id)
def test_representative_manifest_only_fallback_forms_validate(inst):
    assert validation_errors(inst) == []


_THREE_REGS = (RegisterOperand("%r0"), RegisterOperand("%r1"), RegisterOperand("%r2"))
_TWO_REGS = (RegisterOperand("%r0"), RegisterOperand("%r1"))

_MISSING_REQUIRED_MODIFIER_CASES = [
    Instruction("add", (), _THREE_REGS),
    Instruction("ld", (), _TWO_REGS),
    Instruction("cvt", (), _TWO_REGS),
]


@pytest.mark.parametrize("inst", _MISSING_REQUIRED_MODIFIER_CASES, ids=lambda i: i.opcode)
def test_manifest_fallback_does_not_mask_missing_required_modifiers(inst):
    assert any(
        "Missing required modifier" in issue.message
        for issue in validation_errors(inst)
    )
