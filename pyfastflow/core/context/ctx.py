"""
CtxProbe: a standalone, inspectable stand-in for the `ctx` a template
receives, defining - structurally, not by execution - the chain grammar both
template surfaces target.

A template is written as `def tmpl(ctx, i): return ctx.grad(ctx.z.get(i), i)`
(python) or as CUDA text with `$ctx....$` spans (cupy). Both spellings name
the same shape: a dotted attribute path rooted at `ctx`. `ctx.z.get(i)` reads
slot `z`, `ctx.grad(...)` calls slot `grad`, `ctx.grid.neighbour(i, k)` calls
member `neighbour` of composed slot `grid`. Every reference through ctx is
part of the contract - whether or not it is ever called; a bare, uncalled
`ctx.grid.nx` counts exactly as much as a called `ctx.z.get(i)` (see
contract.py's module docstring). Nothing after a call is part of the chain -
the grammar has no case for chaining off a call's return value, and neither
extractor looks for one.

Contract derivation itself (contract.py) never runs a template to find this
out: a Taichi template cannot be called outside kernel-trace context, so the
python surface is read statically, via an AST walk over the template's own
source, and the cupy surface is read directly off its already-materialised
`$...$` span text. CtxProbe is not part of that pipeline - it exists so the
grammar the two extractors agree on has one place, testable by driving a probe
by hand with no template, no AST and no backend involved: `CtxProbe().grid.
neighbour(1, 2)` records the chain `("grid", "neighbour")` in `.touched`,
exactly the shape `extract_python_contract`/`extract_cupy_contract` would
derive from source spelling the same access - and, matching the extractors'
"maximal chain only" rule, a probe never keeps both a chain and a longer one
that extends it: reaching `ctx.z.get` after having only touched `ctx.z` drops
the shorter entry, since `ctx.z.get` is what was actually referenced by the
time attribute access stopped extending it.

CTX_PARAM_NAME is the literal name a python template's first parameter must
carry - see contract.py's `extract_python_contract`, which enforces this.
There is no cupy equivalent to enforce: a `$ctx....$` span already says which
name it means in its own text.

`ctx.bk` (RESERVED_BK_NAME, bk.py) is a second piece of reserved grammar, on
the closure (Taichi/Quadrants) python surface only: the backend-intrinsics
namespace (`ctx.bk.sqrt(x)`, ...), recognised structurally by contract.py and
never a slot a template's Contract requires satisfied - see bk.py's module
docstring for the full mechanism and why it exists.

Author: B.G (08/2026)
"""

CTX_PARAM_NAME = "ctx"
"""The reserved first-parameter name every python template must use - see
extract_python_contract in contract.py."""


class _ChainNode:
    """
    One in-progress dotted path rooted at a CtxProbe, produced by attribute
    access on the probe or on another node.

    Purely structural: `.path` is the tuple of segments walked so far,
    `.dotted` the same joined with ".". Every attribute access - not just a
    terminating call - records into the owning probe's `.touched` (see the
    module docstring); calling a node records the same, already-recorded
    path again, which is a no-op.

    Author: B.G (08/2026)
    """

    def __init__(self, probe: "CtxProbe", path: tuple[str, ...]):
        self._probe = probe
        self._path = path

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    @property
    def dotted(self) -> str:
        return ".".join(self._path)

    def __getattr__(self, name: str) -> "_ChainNode":
        if name.startswith("_"):
            raise AttributeError(name)
        child = _ChainNode(self._probe, self._path + (name,))
        self._probe._record(child._path)
        return child

    def __call__(self, *args, **kwargs) -> "_ChainNode":
        self._probe._record(self._path)
        return self

    def __repr__(self) -> str:
        return f"<ctx.{self.dotted}>"


class CtxProbe:
    """
    The `ctx` placeholder, as a real inspectable object rather than only a
    static-analysis convention. See the module docstring.

    `.touched` accumulates every maximal chain reached so far - the same set
    shape (a set of segment tuples) Contract (contract.py) holds, which is
    the point: a probe can be driven by hand to sanity-check the grammar
    independently of any AST walk or span scan. "Maximal" is enforced on
    every record: a newly recorded chain evicts any already-touched chain it
    strictly extends, and is itself dropped instead of recorded if some
    already-touched chain already extends it (attribute access happening in
    left-to-right order means that second case is rare in practice, but
    cheap to guard regardless).

    Author: B.G (08/2026)
    """

    def __init__(self):
        self.touched: set[tuple[str, ...]] = set()

    def _record(self, path: tuple[str, ...]) -> None:
        if not path:
            return
        if any(len(existing) > len(path) and existing[: len(path)] == path for existing in self.touched):
            return
        for existing in [e for e in self.touched if len(e) < len(path) and path[: len(e)] == e]:
            self.touched.discard(existing)
        self.touched.add(path)

    def __getattr__(self, name: str) -> _ChainNode:
        if name.startswith("_") or name == "touched":
            raise AttributeError(name)
        node = _ChainNode(self, (name,))
        self._record(node._path)
        return node

    def __repr__(self) -> str:
        return f"CtxProbe(touched={sorted('.'.join(c) for c in self.touched)})"
