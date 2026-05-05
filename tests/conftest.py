"""Shared test fixtures."""

from pathlib import Path

from pyptx.ir.nodes import Instruction
from pyptx.spec import validate_instruction

CORPUS_DIR = Path(__file__).parent / "corpus"


def corpus_files() -> list[Path]:
    """Return all .ptx files in the test corpus, sorted by name."""
    return sorted(CORPUS_DIR.glob("*.ptx"))


def validation_errors(inst: Instruction):
    """Return only the error-severity issues from validating ``inst``."""
    return [i for i in validate_instruction(inst) if i.severity == "error"]


def case_id(inst: Instruction) -> str:
    """Build a readable pytest id for an Instruction case."""
    return inst.opcode + "".join(inst.modifiers)
