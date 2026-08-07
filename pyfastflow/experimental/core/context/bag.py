"""
Bag: a named collection of anything a template might bind, plus the operators
that reshape one.

A Bag is a container and nothing more. It never inspects what it holds and has
no notion of backend, mode or compilation - each member is resolved on its own
type by whatever consumes the Bag. That is why this module stands on its own:
it depends on nothing else here beyond the shared uid counter.

merge/extract/trim/replace all return a fresh Bag holding the very
same member objects - no device storage is ever copied, and the same Parameter
reachable from two bags is one Parameter. check_handles is the guard against
the one thing that aliasing can get wrong: a single name meaning two different
objects across the units of one compile.

Author: B.G (07/2026)
"""

from typing import Any

from ..pool.base import new_uid


class Bag:
    """
    A named collection that can be handed to a builder in one go.

    A bag holds whatever a template might want to reach under one name -
    Parameters, Helpers, further Bags, plain python values - mixed
    freely. Nothing dispatches on what a bag contains: each member is resolved
    on its own type when the template is specialized, so a bag grouping a
    quantity with the helpers that act on it works exactly like one holding
    parameters alone.

    Bind it whole - bind("grid", bag) - and reach its members by dotted path
    in the template body (grid.nx.get(i), grid.nbr(i)); or, at the
    RoutineBuilder/SequenceBuilder layer (routine.py/sequence.py),
    bind_bag(bag) sets the one bag every step/block is rebound against at
    compile time.

    Build it, grow it, bind it. There is no removal or reassignment: to change
    the contents, build another bag.

    Author: B.G (07/2026)
    """

    def __init__(self, items: dict[str, Any] | None = None):
        self._uid = new_uid()
        self._items: dict[str, Any] = {}
        for name, item in (items or {}).items():
            self.add(name, item)

    @property
    def uid(self) -> int:
        """
        Process-wide identity assigned at construction, from the same counter
        as Parameters, Helpers and pool data handles. See Parameter.uid.

        Author: B.G (07/2026)
        """
        return self._uid

    def add(self, name: str, item: Any) -> None:
        """
        Register `item` under `name`.

        Parameters
        ----------
        name : str
            Key to register `item` under. Must not already be taken.
        item : Any
            Value to store - a Parameter, Helper, nested Bag or plain value.

        Raises
        ------
        KeyError
            If `name` is already registered.

        Author: B.G (07/2026)
        """
        if name in self._items:
            raise KeyError(f"'{name}' is already registered in this bag")
        self._items[name] = item

    def __getattr__(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()

    def __repr__(self) -> str:
        """
        Every member on its own line at its dotted path, nested Bags shown as
        the subtree they head rather than as an opaque entry.

        Bags are routinely built by merging several others, at which point the
        only reliable way to see what one holds is to read it out; this is
        that. Each leaf is labelled by what it is - a Parameter by mode and
        dtype, anything else by its class - and by its uid, which is what
        makes an alias visible: one object reached under two names shows the
        same uid twice.

        Author: B.G (07/2026)
        """
        lines = [f"Bag(uid={self._uid})"]
        for handle, obj in self.walk():
            if isinstance(obj, Bag):
                lines.append(f"  {handle}/")
                continue
            mode = getattr(obj, "mode", None)
            if mode is not None:
                what = f"{mode} {getattr(obj, 'dtype', '?')}"
            else:
                what = type(obj).__name__
            uid = _uid_of(obj)
            lines.append(f"  {handle}: {what}" + (f" [uid {uid}]" if uid is not None else ""))
        return "\n".join(lines)

    def walk(self, prefix: str = ""):
        """
        Yield (dotted_handle, obj) for every member, descending into nested
        Bags depth-first.

        A nested Bag produces two things: an entry for the Bag itself, at its
        own dotted path, then one entry per member underneath it. So
        `Bag({"at": Bag({"i": p1, "j": p2}), "r": p3})` walks as
        `("at", <Bag>)`, `("at.i", p1)`, `("at.j", p2)`, `("r", p3)` - the
        parent Bag's entry always precedes its members'.

        Parameters
        ----------
        prefix : str, optional
            Dotted path prepended to every yielded handle. Used internally
            for recursion; callers normally leave it at "".

        Returns
        -------
        Iterator[tuple[str, Any]]
            (dotted_handle, obj) pairs in depth-first order.

        Author: B.G (07/2026)
        """
        for name, item in self._items.items():
            handle = f"{prefix}.{name}" if prefix else name
            if isinstance(item, Bag):
                yield handle, item
                yield from item.walk(handle)
            else:
                yield handle, item


def _uid_of(obj: Any) -> int | None:
    """
    An object's uid if it has one, else None.

    Handles bound without a uid (plain python values, unwrapped bindings) are
    simply skipped by check_handles rather than treated as a conflict.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    int or None
        `obj.uid` if it is an int, else None.

    Author: B.G (07/2026)
    """
    uid = getattr(obj, "uid", None)
    return uid if isinstance(uid, int) else None


def check_handles(units: dict[str, dict[str, Any]]) -> None:
    """
    Verify that a handle means the same object everywhere it is used.

    `units` maps a unit name (a kernel, a routine step - whatever the caller
    is checking) to that unit's own {handle: obj} map, typically built from
    Bag.walk(). Across every unit given, the same handle string must resolve
    to objects sharing one uid; if two units bind the same handle to objects
    with different uids, this raises naming the handle and both owning units.

    The converse is fine and common: two different handles pointing at the
    same uid (an alias, or one Parameter reused under two names) is not a
    conflict and is not reported.

    Objects with no `uid` attribute are ignored - there is nothing to compare.

    Parameters
    ----------
    units : dict[str, dict[str, Any]]
        Unit name -> {handle: obj} map, typically each built from a Bag's
        `.walk()`.

    Raises
    ------
    ValueError
        If the same handle resolves to objects with different uids in two
        units, naming the handle and both owning units.

    Author: B.G (07/2026)
    """
    seen: dict[str, tuple[int, str]] = {}
    for unit_name, handles in units.items():
        for handle, obj in handles.items():
            uid = _uid_of(obj)
            if uid is None:
                continue
            prior = seen.get(handle)
            if prior is None:
                seen[handle] = (uid, unit_name)
            elif prior[0] != uid:
                raise ValueError(
                    f"handle '{handle}' is bound to different objects: "
                    f"uid {prior[0]} in '{prior[1]}' vs uid {uid} in '{unit_name}'"
                )


def _resolve_path(bag: "Bag", path: str) -> Any:
    """
    Walk a dotted path through nested Bags and return what it names.

    Parameters
    ----------
    bag : Bag
        Bag to resolve `path` in.
    path : str
        Dotted path, e.g. "at.i".

    Returns
    -------
    Any
        The object named by `path`.

    Raises
    ------
    KeyError
        If any segment is missing or a non-terminal segment does not resolve
        to a Bag, naming the exact prefix that failed.

    Author: B.G (07/2026)
    """
    obj = bag
    parts = path.split(".")
    for depth, part in enumerate(parts):
        if not isinstance(obj, Bag) or part not in obj:
            failed = ".".join(parts[: depth + 1])
            raise KeyError(f"'{path}' not found in bag (no '{failed}')")
        obj = obj[part]
    return obj


def merge(*bags: "Bag") -> "Bag":
    """
    Union of every member across `bags`, into one new Bag.

    Members are taken in argument order; nesting is kept rather than
    flattened, so where two bags carry a Bag under the same name, those two
    are merged recursively instead of one replacing the other.

    A same-name collision between two non-Bag members is allowed silently
    when both share a uid - the same object reached through two bags - and
    raises when they don't, naming the member and both uids. A collision
    where either side has no uid (a plain python value) cannot be resolved
    this way and always raises, since there is nothing to compare.

    No input bag is read from twice or mutated; the result is a fresh Bag.

    Parameters
    ----------
    *bags : Bag
        Bags to union, in precedence order.

    Returns
    -------
    Bag
        Fresh Bag holding the union of all members.

    Raises
    ------
    ValueError
        If a name collides between bags and cannot be proven to be the same
        object.

    Author: B.G (07/2026)
    """
    merged: dict[str, Any] = {}
    for bag in bags:
        for name, item in bag.items():
            if name not in merged:
                merged[name] = item
                continue
            existing = merged[name]
            if isinstance(existing, Bag) and isinstance(item, Bag):
                merged[name] = merge(existing, item)
                continue
            euid, iuid = _uid_of(existing), _uid_of(item)
            if euid is None or iuid is None:
                raise ValueError(
                    f"merge: '{name}' collides between bags and at least one side has "
                    f"no uid to compare, so they cannot be proven to be the same object"
                )
            if euid != iuid:
                raise ValueError(f"merge: '{name}' collides between bags: uid {euid} vs uid {iuid}")
    return Bag(merged)


def extract(bag: "Bag", names) -> "Bag":
    """
    A new Bag holding just the named members of `bag`.

    Each entry in `names` may be a plain name or a dotted path
    (`"stove.at.i"`); a dotted path is resolved through nested Bags and
    reconstructed as nesting in the result, so extracting `"at.i"` and
    `"at.j"` yields a result with an `at` sub-bag holding `i` and `j`, not
    two flat members.

    Parameters
    ----------
    bag : Bag
        Bag to extract from.
    names : Iterable[str]
        Plain names or dotted paths to keep.

    Returns
    -------
    Bag
        Fresh Bag holding just the named members, with nesting rebuilt.

    Raises
    ------
    KeyError
        If any path does not resolve in `bag`.

    Author: B.G (07/2026)
    """
    tree: dict[str, Any] = {}
    for path in names:
        resolved = _resolve_path(bag, path)
        parts = path.split(".")
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = resolved
    return _tree_to_bag(tree)


def _tree_to_bag(tree: dict[str, Any]) -> "Bag":
    """
    Convert the nested-dict scaffolding built by extract()/trim() into
    actual Bags, leaves left untouched.

    Author: B.G (07/2026)
    """
    result = Bag()
    for name, value in tree.items():
        result.add(name, _tree_to_bag(value) if isinstance(value, dict) else value)
    return result


def trim(bag: "Bag", names) -> "Bag":
    """
    `bag` minus the named members, as a new Bag.

    Accepts the same plain-name or dotted-path entries as extract(). Removing
    `"at.i"` drops just that member, leaving `at` in the result with whatever
    else it held; removing a bare name drops that member (and, if it names a
    nested Bag, everything under it) whole.

    Parameters
    ----------
    bag : Bag
        Bag to trim.
    names : Iterable[str]
        Plain names or dotted paths to remove.

    Returns
    -------
    Bag
        Fresh Bag with the named members removed.

    Raises
    ------
    KeyError
        If any path does not resolve in `bag`.

    Author: B.G (07/2026)
    """
    removal: dict[str, Any] = {}
    for path in names:
        _resolve_path(bag, path)  # validates the path exists; raises otherwise
        parts = path.split(".")
        cursor = removal
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = None

    def _copy_minus(b: "Bag", rem: dict[str, Any]) -> "Bag":
        result = Bag()
        for name, item in b.items():
            if name not in rem:
                result.add(name, item)
                continue
            sub = rem[name]
            if sub is None:
                continue
            if not isinstance(item, Bag):
                raise KeyError(f"trim: cannot descend into '{name}': not a Bag")
            result.add(name, _copy_minus(item, sub))
        return result

    return _copy_minus(bag, removal)


def replace(bag: "Bag", name: str, obj: Any) -> "Bag":
    """
    `bag` with the member at `name` swapped for `obj`, as a new Bag.

    This is how anything fixed at a Parameter's construction is changed - its
    mode (see Parameter.mode), or a const's value (see Parameter.set). Both
    mean building a new Parameter, replacing it in here, and recompiling
    whatever bound the old one.

    Parameters
    ----------
    bag : Bag
        Bag to modify.
    name : str
        Plain name or dotted path of the member to replace.
    obj : Any
        Replacement value.

    Returns
    -------
    Bag
        Fresh Bag with `obj` at `name` in place of the old member.

    Raises
    ------
    KeyError
        If `name` does not resolve in `bag`.

    Author: B.G (07/2026)
    """
    parts = name.split(".")

    def _rebuild(b: "Bag", remaining: list[str]) -> "Bag":
        head = remaining[0]
        if head not in b:
            raise KeyError(f"'{name}' not found in bag (no '{head}')")
        result = Bag()
        for iname, item in b.items():
            if iname != head:
                result.add(iname, item)
                continue
            if len(remaining) == 1:
                result.add(iname, obj)
            else:
                if not isinstance(item, Bag):
                    raise KeyError(f"replace: cannot descend into '{head}': not a Bag")
                result.add(iname, _rebuild(item, remaining[1:]))
        return result

    return _rebuild(bag, parts)
