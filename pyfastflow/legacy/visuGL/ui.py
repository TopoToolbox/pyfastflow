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
    def __init__(self, title: str, dock: str = "right", collapsed: bool = False):
        self.title = title
        self.dock = dock
        self.collapsed = collapsed
        self._first_draw = True
        self._items: List[Tuple[str, tuple]] = []

    # Section management ---------------------------------------------------
    def begin_collapsing_header(self, label: str, default_open: bool = False) -> None:
        """Start a collapsible section within the panel."""
        self._items.append(("collapsing_header_begin", (label, default_open)))

    def end_collapsing_header(self) -> None:
        """End a collapsible section."""
        self._items.append(("collapsing_header_end", ()))

    # Primitives -----------------------------------------------------------
    def slider(self, label: str, value: float, vmin: float, vmax: float, log: bool = False, input_field: bool = False) -> ValueRef:
        ref = ValueRef(float(value))
        self._items.append(("slider", (label, ref, float(vmin), float(vmax), bool(log), bool(input_field))))
        return ref

    def int_slider(self, label: str, value: int, vmin: int, vmax: int, input_field: bool = False) -> ValueRef:
        ref = ValueRef(int(value))
        self._items.append(("int_slider", (label, ref, int(vmin), int(vmax), bool(input_field))))
        return ref

    def input_float(self, label: str, ref: ValueRef) -> None:
        """Add a float input field linked to an existing ValueRef."""
        self._items.append(("input_float", (label, ref)))

    def input_int(self, label: str, ref: ValueRef) -> None:
        """Add an integer input field linked to an existing ValueRef."""
        self._items.append(("input_int", (label, ref)))

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
        # Set collapsed state on first draw if requested
        if self._first_draw and self.collapsed:
            try:
                imgui.set_next_window_collapsed(True)
            except Exception:
                pass
            self._first_draw = False
        imgui.begin(self.title)

        # Track collapsing header nesting
        header_stack = []

        for kind, args in list(self._items):
            # Check if we're inside a collapsed header
            if header_stack and not header_stack[-1]:
                # Skip items inside collapsed sections
                if kind == "collapsing_header_end":
                    header_stack.pop()
                continue

            if kind == "collapsing_header_begin":
                label, default_open = args
                # Set default state on first draw
                if self._first_draw:
                    try:
                        imgui.set_next_item_open(default_open)
                    except Exception:
                        pass
                is_open = imgui.collapsing_header(label)[0]
                header_stack.append(is_open)
            elif kind == "collapsing_header_end":
                if header_stack:
                    header_stack.pop()
            elif kind == "slider":
                label, ref, vmin, vmax, log, input_field = args
                val = float(ref.value)
                changed, new_val = imgui.slider_float(label, val, vmin, vmax, '%.3f', 1.0 if not log else 0.0)
                if changed:
                    ref.value = float(new_val)
                # Add input field on same line if requested
                if input_field:
                    imgui.same_line()
                    imgui.push_item_width(100)  # Fixed width for input field
                    # Use current ref.value (which might have just been updated by slider)
                    changed_input, new_input = imgui.input_float(f"##{label}_input", float(ref.value), 0.0, 0.0, '%.3f')
                    imgui.pop_item_width()
                    if changed_input:
                        # Clamp to slider range
                        ref.value = float(max(vmin, min(vmax, new_input)))
            elif kind == "int_slider":
                label, ref, vmin, vmax, input_field = args
                val = int(ref.value)
                changed, new_val = imgui.slider_int(label, val, vmin, vmax)
                if changed:
                    ref.value = int(new_val)
                # Add input field on same line if requested
                if input_field:
                    imgui.same_line()
                    imgui.push_item_width(100)  # Fixed width for input field
                    # Use current ref.value (which might have just been updated by slider)
                    changed_input, new_input = imgui.input_int(f"##{label}_input", int(ref.value))
                    imgui.pop_item_width()
                    if changed_input:
                        # Clamp to slider range
                        ref.value = int(max(vmin, min(vmax, new_input)))
            elif kind == "input_float":
                label, ref = args
                val = float(ref.value)
                changed, new_val = imgui.input_float(label, val, 0.0, 0.0, '%.3f')
                if changed:
                    ref.value = float(new_val)
            elif kind == "input_int":
                label, ref = args
                val = int(ref.value)
                changed, new_val = imgui.input_int(label, val)
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

    def add_panel(self, title: str, dock: str = "right", collapsed: bool = False) -> Panel:
        p = Panel(title, dock, collapsed)
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
