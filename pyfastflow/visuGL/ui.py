from __future__ import annotations

from typing import Callable, List, Tuple, Any


class ValueRef:
    """Tiny holder with .value and .subscribe(fn)."""

    def __init__(self, value: Any):
        self._value = value
        self._subs: List[Callable[[Any], None]] = []

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        self._value = v
        for fn in list(self._subs):
            try:
                fn(v)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[Any], None]) -> None:
        self._subs.append(fn)


class Panel:
    def __init__(self, title: str, dock: str = "right"):
        self.title = title
        self.dock = dock
        self._items: List[Tuple[str, tuple]] = []

    # Primitives -----------------------------------------------------------
    def slider(self, label: str, value: float, vmin: float, vmax: float, log: bool = False) -> ValueRef:
        ref = ValueRef(float(value))
        self._items.append(("slider", (label, ref, float(vmin), float(vmax), bool(log))))
        return ref

    def int_slider(self, label: str, value: int, vmin: int, vmax: int) -> ValueRef:
        ref = ValueRef(int(value))
        self._items.append(("int_slider", (label, ref, int(vmin), int(vmax))))
        return ref

    def checkbox(self, label: str, value: bool = False) -> ValueRef:
        ref = ValueRef(bool(value))
        self._items.append(("checkbox", (label, ref)))
        return ref

    def button(self, label: str, on_click: Callable[[], None]) -> None:
        self._items.append(("button", (label, on_click)))

    # Internal draw --------------------------------------------------------
    def _draw(self, imgui, dock_id=None):
        # Simpler: rely on ImGui's saved docking data; avoid forcing dock id
        # This mirrors the working example's behavior
        imgui.begin(self.title)
        for kind, args in list(self._items):
            if kind == "slider":
                label, ref, vmin, vmax, log = args
                val = float(ref.value)
                changed, new_val = imgui.slider_float(label, val, vmin, vmax, '%.3f', 1.0 if not log else 0.0)
                if changed:
                    ref.value = float(new_val)
            elif kind == "int_slider":
                label, ref, vmin, vmax = args
                val = int(ref.value)
                changed, new_val = imgui.slider_int(label, val, vmin, vmax)
                if changed:
                    ref.value = int(new_val)
            elif kind == "checkbox":
                label, ref = args
                changed, new_val = imgui.checkbox(label, bool(ref.value))
                if changed:
                    ref.value = bool(new_val)
            elif kind == "button":
                label, cb = args
                if imgui.button(label):
                    try:
                        cb()
                    except Exception:
                        pass
        imgui.end()


class UI:
    def __init__(self):
        self._panels: List[Panel] = []
        self._docking = False

    def enable_docking(self, flag: bool) -> None:
        self._docking = bool(flag)

    def add_panel(self, title: str, dock: str = "right") -> Panel:
        p = Panel(title, dock)
        self._panels.append(p)
        return p

    # Internal
    def _draw_all_panels(self, imgui):
        if self._docking:
            try:
                vp = imgui.get_main_viewport()
                imgui.set_next_window_pos(vp.pos.x, vp.pos.y)
                imgui.set_next_window_size(vp.size.x, vp.size.y)
                imgui.set_next_window_viewport(vp.id)
                flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_COLLAPSE |
                         imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
                         imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
                         imgui.WINDOW_NO_NAV_FOCUS | imgui.WINDOW_MENU_BAR)
                imgui.begin("DockSpace", True, flags)
                imgui.dock_space(imgui.get_id("MainDock"), 0.0, 0.0, 0)
                imgui.end()
            except Exception:
                pass
        for p in list(self._panels):
            p._draw(imgui)
