"""
A composite's structural contract - the set of ctx.* chains its template
actually touches - derived, never hand-authored, from the template's own
source.

Two extractors read that source and produce the same kind of thing, a
Contract: a frozen set of chains, each chain the maximal tuple of dotted
segments following `ctx` - every reference through ctx counts, whether or not
it is ever called. `ctx.z.get(i)` and a hypothetical bare `ctx.grid.nx` used
as a value both contribute a chain; nothing about the grammar privileges a
call over any other use. What a chain's own trailing segment means for
device emission - is it a required `.get`, is a bare non-call reference even
legal - is a backend question, deliberately not decided here (see slot.py's
module docstring on PARAM access being uniform across modes, and 1c for
where "is this spelling legal" actually gets enforced).

extract_python_contract(template)   a python def, read by inspect.getsource +
                                     ast - STATIC ANALYSIS, the template is
                                     never called. A Taichi/Quadrants template
                                     cannot run outside kernel-trace context,
                                     so this is the only sound way to find out
                                     what it touches, and it is also why `ctx`
                                     must appear only in the template body
                                     itself - a lambda, an exec'd function, or
                                     `ctx` passed into a nested python def the
                                     walk cannot see through, all raise
                                     ContractError rather than silently
                                     under-reporting the contract.
extract_cupy_contract(source)       CUDA text already carrying `$...$` spans.
                                     The template is an f-string fully
                                     materialised at build time, so this reads
                                     the final string directly - no
                                     inspect.getsource, and none of the
                                     python surface's restrictions apply,
                                     since there is no python callable to lose
                                     sight of in the first place.

Contract.check_root(root, provided) is the candidate-check compose() (builder.
py) runs once a template's contract is known: for every chain this contract
requires under `root` (e.g. `grid.neighbour`), the composed candidate must
provide the next segment (`neighbour`) among its own top-level names, or this
raises naming exactly what is missing and what the candidate offers instead.

Author: B.G (08/2026)
"""

import ast
import inspect
import re
import textwrap
from typing import Callable

from .ctx import CTX_PARAM_NAME

Chain = tuple[str, ...]


class ContractError(Exception):
    """
    Raised when a template's source cannot be turned into a contract (no
    recoverable source, `ctx` not the first parameter, a malformed span), or
    when a derived contract is checked against a candidate that does not
    satisfy it.

    Author: B.G (08/2026)
    """


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


class Contract:
    """
    A composite's derived structural contract: the frozen set of ctx.* chains
    its template touches. See the module docstring.

    Author: B.G (08/2026)
    """

    def __init__(self, chains: frozenset[Chain]):
        self._chains = frozenset(chains)

    @property
    def chains(self) -> frozenset[Chain]:
        """Every chain this contract requires, as a segment tuple each."""
        return self._chains

    @property
    def roots(self) -> set[str]:
        """The first segment of every chain - the ctx.* names this contract references directly."""
        return {chain[0] for chain in self._chains if chain}

    def check_root(self, root: str, provided: set[str]) -> None:
        """
        Verify a composed candidate for slot `root` satisfies every chain
        this contract requires under it.

        A chain `("grid", "neighbour")` requires `"neighbour"` to be among
        `provided` - the candidate's own top-level names (see frozen.py,
        `_Frozen.provides`). Raises ContractError naming the first missing
        member (and how many more, if any) and what the candidate provides,
        if any chain's next segment is absent. A chain of length 1 rooted at
        `root` (bare `ctx.root`, no further member) needs nothing from
        `provided` - the root itself being composed is enough.

        Author: B.G (08/2026)
        """
        missing = sorted(
            {chain[1] for chain in self._chains if len(chain) > 1 and chain[0] == root and chain[1] not in provided}
        )
        if not missing:
            return
        extra = f" (+{len(missing) - 1} more: {', '.join(missing[1:])})" if len(missing) > 1 else ""
        raise ContractError(
            f"requires {root}.{missing[0]}{extra}, candidate provides {sorted(provided)}"
        )

    def __repr__(self) -> str:
        if not self._chains:
            return "Contract()"
        body = ", ".join("ctx." + ".".join(c) for c in sorted(self._chains))
        return f"Contract({body})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Contract) and self._chains == other._chains

    def __hash__(self) -> int:
        return hash(self._chains)


# ---------------------------------------------------------------------------
# python surface: static AST walk
# ---------------------------------------------------------------------------


def _ctx_chain(node: ast.AST) -> Chain | None:
    """
    If `node` is an Attribute/Name chain rooted at `ctx`, its segments in
    source order (`ctx.grid.neighbour` -> `("grid", "neighbour")`); else None.

    Author: B.G (08/2026)
    """
    segments: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        segments.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == CTX_PARAM_NAME:
        segments.reverse()
        return tuple(segments) if segments else None
    return None


class _ChainVisitor(ast.NodeVisitor):
    """
    Collects every maximal ctx.* chain in a template body, called or not.

    Only `visit_Attribute` is overridden: when a node's own chain resolves
    all the way down to `ctx` (see `_ctx_chain`), that is by construction the
    maximal chain at this point in the tree - a `ctx.grid.neighbour` node's
    `.value` is `ctx.grid`, a strict prefix, never a separate reference worth
    recording on its own - so this records the chain and does not descend
    into `node.value`. A node that does not resolve to ctx falls through to
    generic_visit, so a ctx chain nested anywhere within it (a call argument,
    a binary operand, ...) is still found by ordinary recursion. Whether the
    chain is then the func of a Call or used bare as a value makes no
    difference here - both shapes are recorded identically (see the module
    docstring).

    Author: B.G (08/2026)
    """

    def __init__(self):
        self.chains: set[Chain] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _ctx_chain(node)
        if chain is not None:
            self.chains.add(chain)
            return
        self.generic_visit(node)


def _get_function_ast(template: Callable) -> ast.FunctionDef:
    """
    The single FunctionDef node for `template`'s own source, dedented and
    parsed. Raises ContractError - naming the template - if the source
    cannot be recovered (a lambda, an exec'd function) or does not parse down
    to one function definition.

    Author: B.G (08/2026)
    """
    name = getattr(template, "__name__", repr(template))
    try:
        source = inspect.getsource(template)
    except (OSError, TypeError) as exc:
        raise ContractError(
            f"template {name!r}: no recoverable source (a lambda or exec'd function cannot "
            f"be statically analysed - ctx's contract can only be derived from a def with "
            f"real source)"
        ) from exc
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError as exc:
        raise ContractError(f"template {name!r}: source does not parse: {exc}") from exc
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(body) != 1:
        raise ContractError(
            f"template {name!r}: expected source recoverable to exactly one function "
            f"definition, got {len(body)}"
        )
    return body[0]


def extract_python_contract(template: Callable) -> Contract:
    """
    The Contract a python template requires, by static AST walk over its own
    source. Never calls `template` - see the module docstring for why.

    Enforces that `ctx` is the template's first parameter (positional or
    positional-or-keyword) - a template reaching a `ctx` handed to it under
    another name, or not receiving one as its first argument at all, is
    rejected here rather than silently producing an empty or wrong contract.

    Author: B.G (08/2026)
    """
    fn = _get_function_ast(template)
    name = getattr(template, "__name__", fn.name)
    params = fn.args.posonlyargs + fn.args.args
    if not params or params[0].arg != CTX_PARAM_NAME:
        got = params[0].arg if params else "(no parameters)"
        raise ContractError(
            f"template {name!r}: first parameter must be named {CTX_PARAM_NAME!r}, got {got!r}"
        )
    visitor = _ChainVisitor()
    visitor.visit(fn)
    return Contract(frozenset(visitor.chains))


# ---------------------------------------------------------------------------
# cupy surface: span text scan
# ---------------------------------------------------------------------------

_SPAN_RE = re.compile(r"\$(.*?)\$", re.S)
_PATH_RE = re.compile(r"^([\w.]+)")


def extract_cupy_contract(source: str) -> Contract:
    """
    The Contract a cupy (CUDA source text) template requires, by scanning its
    already-materialised `$...$` spans for ones prefixed `ctx.` - see
    cupy_backend.py's _SpanParser for the span mechanism this reads, and the
    module docstring for why this needs no AST and none of the python
    surface's restrictions: the template is a plain string by the time this
    runs, nothing to lose sight of.

    A span not prefixed `ctx.` (a bound plain value, a bare const name used
    outside any span) contributes nothing - only ctx-rooted spans are part of
    this template's structural contract. A span's own trailing `(...)` is not
    part of the recorded chain and its presence or absence makes no
    difference - `$ctx.grid.neighbour(i, k)$` and a hypothetical bare
    `$ctx.grid.neighbour$` both record `("grid", "neighbour")` (see the
    module docstring: every reference through ctx counts, called or not).

    Author: B.G (08/2026)
    """
    chains: set[Chain] = set()
    for match in _SPAN_RE.finditer(source):
        inner = match.group(1).strip()
        path_match = _PATH_RE.match(inner)
        if path_match is None:
            raise ContractError(f"malformed span: ${inner}$")
        parts = path_match.group(1).split(".")
        if parts[0] != CTX_PARAM_NAME:
            continue
        segments = tuple(parts[1:])
        if segments:
            chains.add(segments)
    return Contract(frozenset(chains))
