"""
Bag: a named collection of anything a template might bind.

A Bag is a container and nothing more. It never inspects what it holds and has
no notion of backend, mode or compilation - each member is resolved on its own
type by whatever consumes the Bag. That is why this module stands on its own:
it depends on nothing else here beyond the shared uid counter.

Build it, grow it with `add`, read members back by attribute (`bag.name`),
item (`bag["name"]`) or `walk()`. There is no removal or reassignment and no
operator that reshapes a Bag: to change the contents, build another Bag.

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
    in the template body (grid.nx.get(i), grid.nbr(i)).

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

    Handles bound without a uid (plain python values, unwrapped bindings)
    return None; `__repr__` shows them with no uid rather than as a conflict.

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
