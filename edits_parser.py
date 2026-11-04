"""Parser for the ``edits`` weight language used in the released replies.

The language, as the prompt that produced the replies defines it:

* only the last ```` ```edits ```` block is read; it ends at the next line that is just ```` ``` ````;
* one statement per line, ``#`` starts a comment, blank lines are skipped;
* a list value may continue over following lines until its brackets balance;
* tensors ``W1 b1 W2 b2 W3 b3``; per-axis index forms ``i``, ``a:b``, ``a:b:s``, ``:``;
* operators ``=`` (scalar fill, 1-D list of exactly the selected count, or 2-D list of the block's
  shape, row-major), ``*=``, ``+=``, ``-=`` with a scalar;
* any |value| > 1e6 skips the statement; statements apply in order; a statement that cannot be read
  is skipped and recorded, the others still apply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SHAPES: dict[str, tuple[int, ...]] = {
    "W1": (600, 32), "b1": (32,),
    "W2": (32, 32), "b2": (32,),
    "W3": (32, 6), "b3": (6,),
}
NPZ_KEYS: dict[str, str] = {
    "W1": "actor/dense1/kernel", "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel", "b3": "actor/logits/bias",
}
MAX_ABS = 1e6

_STMT = re.compile(r"^\s*(\w+)\s*\[([^\]]*)\]\s*(\*=|\+=|-=|=)\s*(.*?)\s*$", re.S)
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TOKEN = re.compile(r"\s*(\[|\]|,|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
_INT = re.compile(r"^-?\d+$")
# The head of a statement line: NAME[index] op, followed by the start of the value.
_HEAD = re.compile(r"^\s*\w+\s*\[[^\]]*\]\s*(?:\*=|\+=|-=|=)\s*(.*)$", re.S)


@dataclass
class ParseResult:
    tensors: dict[str, np.ndarray]
    block_found: bool
    cut_off: bool
    statements: int
    applied: int
    errors: list[tuple[int, str, str]] = field(default_factory=list)


class _Bad(Exception):
    """A statement that cannot be read; the message is the recorded reason."""


def _last_block(text: str) -> tuple[list[str] | None, bool]:
    """Lines of the last ```edits block (without the fences) and whether it was left open."""
    block: list[str] | None = None
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if current is None:
            if stripped == "```edits":
                current = []
        elif stripped == "```":
            block, current = current, None
        else:
            current.append(line)
    if current is not None:  # opened, never closed
        return current, True
    return block, False


def _statements(lines: list[str]) -> list[tuple[int, str]]:
    """(1-based line number within the block, statement text), joining continued list values.

    Only a list value may continue over following lines, so bracket depth is counted from the
    right-hand side of a recognizable ``NAME[index] op [`` head.  Any other line, including a
    malformed one such as ``b1[bad = 1``, is a complete single-line statement: it is reported on
    its own and never swallows the statements after it.
    """
    out: list[tuple[int, str]] = []
    start, parts, depth = 0, [], 0
    for n, raw in enumerate(lines, 1):
        code = raw.split("#", 1)[0].strip()
        if not parts:
            if not code:
                continue
            start = n
            head = _HEAD.match(code)
            rhs = head.group(1) if head else ""
            if not rhs.startswith("["):
                out.append((start, code))
                continue
            depth = rhs.count("[") - rhs.count("]")
        else:
            depth += code.count("[") - code.count("]")
        parts.append(code)
        if depth <= 0:
            out.append((start, " ".join(p for p in parts if p)))
            parts, depth = [], 0
    if parts:
        out.append((start, " ".join(p for p in parts if p)))
    return out


def _axis(spec: str, n: int) -> list[int]:
    spec = spec.strip()
    if spec == ":":
        return list(range(n))
    if _INT.match(spec):
        i = int(spec)
        if not 0 <= i < n:
            raise _Bad(f"index {i} out of range for axis of size {n}")
        return [i]
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(_INT.match(p.strip()) for p in parts):
        raise _Bad(f"bad index '{spec}'")
    a, b = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 1
    if s <= 0:
        raise _Bad(f"bad step {s} in '{spec}'")
    return list(range(n))[a:b:s]


def _selection(index: str, shape: tuple[int, ...]) -> tuple[list[int], ...]:
    axes = index.split(",")
    if len(axes) != len(shape):
        raise _Bad(f"{len(axes)} index/indices for a {len(shape)}-D tensor")
    return tuple(_axis(a, n) for a, n in zip(axes, shape))


def _parse_value(text: str):
    """A float, or a (possibly nested) list of floats, from the strict number/list grammar."""
    if _NUMBER.fullmatch(text):
        return float(text)
    pos = 0
    tokens: list[str] = []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise _Bad(f"cannot read value '{text}'")
        tokens.append(m.group(1))
        pos = m.end()
    i = 0

    def item():
        nonlocal i
        if i >= len(tokens):
            raise _Bad("unterminated list")
        tok = tokens[i]
        i += 1
        if tok != "[":
            if tok in ("]", ","):
                raise _Bad(f"unexpected '{tok}' in value")
            return float(tok)
        items = []
        if i < len(tokens) and tokens[i] == "]":
            i += 1
            return items
        while True:
            items.append(item())
            if i >= len(tokens):
                raise _Bad("unterminated list")
            tok = tokens[i]
            i += 1
            if tok == "]":
                return items
            if tok != ",":
                raise _Bad(f"expected ',' or ']' in list, got '{tok}'")

    value = item()
    if i != len(tokens):
        raise _Bad(f"trailing text after value: '{text}'")
    if not isinstance(value, list):
        raise _Bad(f"cannot read value '{text}'")
    return value


def _flat(value) -> list[float]:
    if isinstance(value, list):
        return [x for v in value for x in _flat(v)]
    return [value]


def _apply(stmt: str, tensors: dict[str, np.ndarray]) -> None:
    m = _STMT.match(stmt)
    if not m:
        raise _Bad("not a statement of the form NAME[index] op value")
    name, index, op, rhs = m.groups()
    if name not in tensors:
        raise _Bad(f"unknown tensor '{name}'")
    arr = tensors[name]
    sel = _selection(index, arr.shape)
    value = _parse_value(rhs)
    if any(abs(v) > MAX_ABS for v in _flat(value)):
        raise _Bad(f"a value exceeds {MAX_ABS:g} in magnitude")
    ix = np.ix_(*sel)
    shape = tuple(len(s) for s in sel)
    if op != "=":
        if isinstance(value, list):
            raise _Bad(f"'{op}' takes a scalar")
        scalar = np.float32(value)
        block = arr[ix]
        if op == "*=":
            block *= scalar
        elif op == "+=":
            block += scalar
        else:
            block -= scalar
        arr[ix] = block
        return
    if not isinstance(value, list):
        arr[ix] = np.float32(value)
        return
    count = int(np.prod(shape))
    if all(not isinstance(v, list) for v in value):
        if len(value) != count:
            raise _Bad(f"list has {len(value)} values but {count} entries are selected")
        arr[ix] = np.asarray(value, dtype=np.float32).reshape(shape)
        return
    if len(shape) != 2 or not all(
        isinstance(row, list) and all(not isinstance(v, list) for v in row) for row in value
    ):
        raise _Bad("a 2-D list needs a 2-D block and rows of numbers")
    widths = {len(row) for row in value}
    if len(widths) != 1:
        raise _Bad("2-D list rows differ in length")
    if (len(value), widths.pop()) != shape:
        raise _Bad(f"2-D list shape does not match the selected block {shape}")
    arr[ix] = np.asarray(value, dtype=np.float32)


def parse_reply(text: str) -> ParseResult:
    tensors = {name: np.zeros(shape, dtype=np.float32) for name, shape in SHAPES.items()}
    lines, cut_off = _last_block(text)
    result = ParseResult(tensors, lines is not None, cut_off, 0, 0)
    if lines is None:
        return result
    for lineno, stmt in _statements(lines):
        result.statements += 1
        try:
            _apply(stmt, tensors)
            result.applied += 1
        except _Bad as e:
            result.errors.append((lineno, stmt, str(e)))
    return result


def to_npz(result: ParseResult, path: str) -> None:
    np.savez(path, **{NPZ_KEYS[name]: result.tensors[name] for name in SHAPES})
