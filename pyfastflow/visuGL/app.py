from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import imgui
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from .scene import Scene
from .ui import UI
from .data import DataHub


@dataclass
class OrbitCamera:
    yaw: float = 0.0
    pitch: float = 30.0
    distance: float = 5.0
    # Use default_factory for mutable default
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))

    def get_view_matrix(self) -> np.ndarray:
        pr = np.radians(self.pitch)
        yr = np.radians(self.yaw)
        x = self.distance * np.cos(pr) * np.cos(yr)
        z = self.distance * np.cos(pr) * np.sin(yr)
        y = self.distance * np.sin(pr)
        eye = self.target + np.array([x, y, z], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(eye, self.target, up)

    def get_eye(self) -> np.ndarray:
        pr = np.radians(self.pitch)
        yr = np.radians(self.yaw)
        x = self.distance * np.cos(pr) * np.cos(yr)
        z = self.distance * np.cos(pr) * np.sin(yr)
        y = self.distance * np.sin(pr)
        return self.target + np.array([x, y, z], dtype=np.float32)

    # Interaction helpers --------------------------------------------------
    def orbit(self, dx: float, dy: float, sensitivity: float = 0.5) -> None:
        self.yaw += dx * sensitivity
        self.pitch = float(np.clip(self.pitch - dy * sensitivity, -89.0, 89.0))

    def pan(self, dx: float, dy: float) -> None:
        pr = np.radians(self.pitch)
        yr = np.radians(self.yaw)
        right = np.array([-np.sin(yr), 0.0, np.cos(yr)], dtype=np.float32)
        forward = np.array([np.cos(pr) * np.cos(yr), np.sin(pr), np.cos(pr) * np.sin(yr)], dtype=np.float32)
        right /= max(1e-6, np.linalg.norm(right))
        forward /= max(1e-6, np.linalg.norm(forward))
        up = np.cross(right, forward)
        up /= max(1e-6, np.linalg.norm(up))
        pan_speed = 0.01 * self.distance
        self.target = self.target + right * dx * pan_speed + up * (-dy) * pan_speed

    def dolly(self, yoffset: float) -> None:
        zoom_speed = 0.1
        self.distance = float(max(0.5, self.distance - yoffset * zoom_speed * self.distance))


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = (target - eye).astype(np.float32)
    f /= max(1e-6, np.linalg.norm(f))
    s = np.cross(f, up)
    s /= max(1e-6, np.linalg.norm(s))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(1e-6, aspect)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


class GLApp:
    """GLFW + ModernGL + ImGui (docking) wrapper with tiny surface.

    - scene: ordered list of Layers
    - ui: panels with minimal primitives
    - data: GPU uploads hub
    - camera: OrbitCamera
    """

    def __init__(self, title: str = "PyFastFlow Viewer") -> None:
        self.title = title
        self.scene = Scene()
        self.ui = UI()
        self.data = DataHub()
        self.camera = OrbitCamera()
        self._init_cbs: List[Callable[[GLApp], None]] = []
        self._frame_cbs: List[Callable[[float], None]] = []
        self._gui_cbs: List[Callable[[], None]] = []
        self._fps_overlay = True

    # Hooks ---------------------------------------------------------------
    def on_init(self, fn: Callable[["GLApp"], None]) -> None:
        self._init_cbs.append(fn)

    def on_frame(self, fn: Callable[[float], None]) -> None:
        self._frame_cbs.append(fn)

    def on_gui(self, fn: Callable[[], None]) -> None:
        self._gui_cbs.append(fn)

    # Runtime -------------------------------------------------------------
    def run(self) -> None:
        # Avoid moderngl_window consuming unrelated CLI args
        argv_backup = list(sys.argv)
        sys.argv = [argv_backup[0]]

        glapp = self  # capture in closure

        class _Cfg(mglw.WindowConfig):
            gl_version = (3, 3)
            title = glapp.title
            window_size = (1200, 800)
            aspect_ratio = None
            resource_dir = '.'
            window = 'glfw'
            vsync = True
            cursor = True

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # ImGui setup
                imgui.create_context()
                self._imgui = ModernglWindowRenderer(self.wnd)
                io = self._imgui.io
                io.config_flags |= int(getattr(imgui, "CONFIG_DOCKING_ENABLE", 0))
                # Initialize display size/scaling for imgui immediately using framebuffer size
                try:
                    ww, wh = self.wnd.size
                except Exception:
                    ww, wh = (1200, 800)
                try:
                    # Renderer expects window size here (matches working example)
                    self._imgui.resize(int(ww), int(wh))
                except Exception:
                    pass
                try:
                    _update_imgui_display(int(ww), int(wh), self.wnd)
                except Exception:
                    pass
                # Inject context into data hub
                glapp.data._set_ctx(self.ctx)
                # Call init hooks
                for fn in glapp._init_cbs:
                    try:
                        fn(glapp)
                    except Exception:
                        pass
                # Setup scene
                glapp.scene._setup(self.ctx, glapp.data)
                self._last = time.perf_counter()
                # Event counters and last-pos tracking (robust inputs)
                self._evt_counts = {"pos": 0, "drag": 0, "press": 0, "release": 0, "scroll": 0}
                self._last_event_pos = (float("nan"), float("nan"))
                self._last_mouse_pos = (float("nan"), float("nan"))
                self._last_evt_dxdy = (0.0, 0.0)
                self.relative_mouse = False

            # Events ------------------------------------------------------
            def key_event(self, key, action, modifiers):  # type: ignore[override]
                try:
                    self._imgui.key_event(key, action, modifiers)
                except Exception:
                    pass
                # ESC to close, F11 to toggle fullscreen
                if action == self.wnd.keys.ACTION_PRESS:
                    if key == self.wnd.keys.ESCAPE:
                        self.wnd.close()
                    if key == self.wnd.keys.F11:
                        try:
                            self.wnd.toggle_fullscreen()
                        except Exception:
                            pass

            def _get_glfw_window(self):
                for name in ('_window', '_native_window', 'window'):
                    try:
                        win = getattr(self.wnd, name, None)
                        if win is not None:
                            return win
                    except Exception:
                        pass
                try:
                    return getattr(getattr(self.wnd, 'ctx', None), 'window', None)
                except Exception:
                    return None

            def _buttons(self):
                ms = getattr(self.wnd, 'mouse_states', None)
                left = bool(getattr(ms, 'left', False)) if ms else False
                right = bool(getattr(ms, 'right', False)) if ms else False
                middle = bool(getattr(ms, 'middle', False)) if ms else False
                if not (left or right or middle):
                    try:
                        import glfw  # type: ignore
                        raw = self._get_glfw_window()
                        if raw is not None:
                            left = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                            right = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                            middle = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                    except Exception:
                        pass
                return left, right, middle

            def mouse_drag_event(self, x, y, dx, dy):  # type: ignore[override]
                try:
                    self._imgui.mouse_drag_event(x, y, dx, dy)
                except Exception:
                    pass
                io = imgui.get_io()
                self._evt_counts["drag"] += 1
                if all(np.isfinite(v) for v in (x, y)):
                    self._last_event_pos = (float(x), float(y))
                if io.want_capture_mouse:
                    return
                l, r, m = self._buttons()
                if l:
                    glapp.camera.orbit(dx, dy)
                elif r or m:
                    glapp.camera.pan(dx, dy)

            def mouse_position_event(self, x, y, dx, dy):  # type: ignore[override]
                try:
                    self._imgui.mouse_position_event(x, y, dx, dy)
                except Exception:
                    pass
                self._evt_counts["pos"] += 1
                if all(np.isfinite(v) for v in (x, y)):
                    self._last_event_pos = (float(x), float(y))
                io = self._imgui.io
                if io.want_capture_mouse:
                    return
                l, r, m = self._buttons()
                if l or r or m:
                    if l:
                        glapp.camera.orbit(dx, dy)
                    else:
                        glapp.camera.pan(dx, dy)

            def mouse_scroll_event(self, x_offset, y_offset):  # type: ignore[override]
                try:
                    self._imgui.mouse_scroll_event(x_offset, y_offset)
                except Exception:
                    pass
                self._evt_counts["scroll"] += 1
                # Zoom to mouse (raycast) like the working demo
                self._apply_zoom_to_mouse(y_offset)

            def mouse_press_event(self, x, y, button):  # type: ignore[override]
                try:
                    self._imgui.mouse_press_event(x, y, button)
                except Exception:
                    pass
                self._evt_counts["press"] += 1

            def mouse_release_event(self, x, y, button):  # type: ignore[override]
                try:
                    self._imgui.mouse_release_event(x, y, button)
                except Exception:
                    pass
                self._evt_counts["release"] += 1

            def resize(self, width: int, height: int):  # type: ignore[override]
                self.ctx.viewport = (0, 0, max(1, width), max(1, height))
                try:
                    self._imgui.resize(int(width), int(height))
                except Exception:
                    pass
                try:
                    _update_imgui_display(width, height, self.wnd)
                except Exception:
                    pass

            def buffer_size_event(self, width: int, height: int):  # type: ignore[override]
                self.ctx.viewport = (0, 0, max(1, width), max(1, height))
                try:
                    # Update only the fb scale in imgui io
                    try:
                        win_w, win_h = self.wnd.size
                    except Exception:
                        win_w, win_h = width, height
                    _update_imgui_display(win_w, win_h, self.wnd)
                except Exception:
                    pass

            # Optional unicode / modifiers forwarding to improve IO
            def unicode_char_entered(self, char: str):  # type: ignore[override]
                try:
                    self._imgui.unicode_char_entered(char)
                except Exception:
                    pass

            def modifiers_event(self, modifiers: int):  # type: ignore[override]
                try:
                    self._imgui.modifiers_event(modifiers)
                except Exception:
                    pass

            # Main loop step ----------------------------------------------
            def render(self, time_s: float, frame_time: float):  # type: ignore[override]
                now = time.perf_counter()
                dt = now - self._last
                self._last = now
                # Model stepping
                for fn in glapp._frame_cbs:
                    try:
                        fn(dt)
                    except Exception:
                        pass

                # Ensure viewport matches framebuffer
                try:
                    fb_w, fb_h = self.wnd.buffer_size
                    self.ctx.viewport = (0, 0, int(fb_w), int(fb_h))
                except Exception:
                    fb_w, fb_h = self.wnd.size
                    self.ctx.viewport = (0, 0, int(fb_w), int(fb_h))

                # Sync ImGui IO like the working example
                try:
                    # Fallback to GLFW cursor only if we received no motion yet
                    if (self._evt_counts["pos"] + self._evt_counts["drag"]) == 0 and 'glfw' in type(self.wnd).__module__:
                        raw_win = self._get_glfw_window()
                        if raw_win is not None:
                            import glfw  # type: ignore
                            cx, cy = glfw.get_cursor_pos(raw_win)
                            if all(np.isfinite(v) for v in (cx, cy)):
                                self._last_event_pos = (float(cx), float(cy))
                                self._last_mouse_pos = (float(cx), float(cy))
                    # Update IO with current sizes and process inputs
                    win_w, win_h = self.wnd.size
                    _update_imgui_display(int(win_w), int(win_h), self.wnd)
                    self._imgui.process_inputs()
                    # Fix invalid mouse pos after processing
                    io_fix = imgui.get_io()
                    mp = tuple(io_fix.mouse_pos)
                    if len(mp) != 2 or not (np.isfinite(mp[0]) and np.isfinite(mp[1])):
                        io_fix.mouse_pos = self._last_event_pos
                    # Also ensure mouse_down reflects real button states
                    try:
                        ms = getattr(self.wnd, 'mouse_states', None)
                        if ms:
                            io_fix.mouse_down[0] = bool(ms.left)
                            io_fix.mouse_down[1] = bool(ms.right)
                            io_fix.mouse_down[2] = bool(ms.middle)
                    except Exception:
                        pass
                except Exception:
                    pass

                # New GUI frame
                imgui.new_frame()
                for fn in glapp._gui_cbs:
                    try:
                        fn()
                    except Exception:
                        pass
                glapp.ui._draw_all_panels(imgui)

                # Clear and draw scene
                try:
                    self.ctx.clear(0.05, 0.05, 0.08, 1.0, depth=1.0)
                except TypeError:
                    self.ctx.clear(0.05, 0.05, 0.08, 1.0)
                # Compute and stash camera matrices based on framebuffer aspect
                aspect = max(1.0, float(fb_w) / max(1.0, float(fb_h)))
                proj = perspective(np.radians(60.0), aspect, 0.1, 100.0)
                view = glapp.camera.get_view_matrix()
                self._last_proj = proj
                self._last_view = view
                glapp.scene.draw_all(self.ctx, glapp.camera)

                # FPS overlay
                if glapp._fps_overlay:
                    imgui.set_next_window_bg_alpha(0.3)
                    imgui.set_next_window_position(10.0, 10.0)
                    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE |
                             imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS)
                    imgui.begin("FPS", True, flags)
                    imgui.text(f"FPS: {1.0/max(1e-6, frame_time):.0f}")
                    if imgui.button("Hide"):
                        glapp._fps_overlay = False
                    imgui.end()

                # Render GUI
                imgui.render()
                self._imgui.render(imgui.get_draw_data())
                # Fallback when motion events are not delivered
                io2 = self._imgui.io
                if not io2.want_capture_mouse and ((self._evt_counts["pos"] + self._evt_counts["drag"]) == 0 or self.relative_mouse):
                    try:
                        import glfw  # type: ignore
                        raw_win = getattr(self.wnd, '_window', None)
                        if raw_win is not None:
                            cx, cy = glfw.get_cursor_pos(raw_win)
                            if all(np.isfinite(v) for v in (cx, cy)) and all(np.isfinite(v) for v in self._last_mouse_pos):
                                dx = float(cx - self._last_mouse_pos[0])
                                dy = float(cy - self._last_mouse_pos[1])
                                self._last_mouse_pos = (float(cx), float(cy))
                                l, r, m = self._buttons()
                                if l:
                                    glapp.camera.orbit(dx, dy)
                                elif r or m:
                                    glapp.camera.pan(dx, dy)
                    except Exception:
                        pass

            # Some backends call on_render instead
            def on_render(self, time: float, frame_time: float):  # type: ignore[override]
                return self.render(time, frame_time)

            # Compatibility for some backends
            def scroll_event(self, x_offset, y_offset):  # type: ignore[override]
                return self.mouse_scroll_event(x_offset, y_offset)

            # Zoom to mouse adapted from working demo
            def _apply_zoom_to_mouse(self, y_offset: float):
                import math
                io = imgui.get_io()
                # Window size
                try:
                    win_w, win_h = io.display_size
                    win_w = float(win_w)
                    win_h = float(win_h)
                except Exception:
                    try:
                        ww, wh = self.wnd.size
                        win_w, win_h = float(ww), float(wh)
                    except Exception:
                        win_w, win_h = 1200.0, 800.0

                # Mouse position
                mp = getattr(io, 'mouse_pos', (-1.0, -1.0))
                mx, my = float(mp[0]), float(mp[1])
                if not (np.isfinite(mx) and np.isfinite(my)):
                    try:
                        mx2, my2 = self.wnd.mouse_position
                        if np.isfinite(mx2) and np.isfinite(my2):
                            mx, my = float(mx2), float(my2)
                    except Exception:
                        mx, my = win_w * 0.5, win_h * 0.5

                # Build matrices
                aspect = max(1.0, win_w / max(1.0, win_h))
                proj = perspective(np.radians(60.0), aspect, 0.1, 100.0)
                view = glapp.camera.get_view_matrix()
                vp = proj @ view
                try:
                    inv_vp = np.linalg.inv(vp)
                except Exception:
                    inv_vp = None

                hit = None
                if inv_vp is not None:
                    # to NDC
                    x_ndc = (2.0 * float(mx) / max(1.0, float(win_w))) - 1.0
                    y_ndc = 1.0 - (2.0 * float(my) / max(1.0, float(win_h)))
                    near_ndc = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float32)
                    far_ndc = np.array([x_ndc, y_ndc, 1.0, 1.0], dtype=np.float32)
                    near_world = inv_vp @ near_ndc
                    far_world = inv_vp @ far_ndc
                    if near_world[3] != 0:
                        near_world /= near_world[3]
                    if far_world[3] != 0:
                        far_world /= far_world[3]
                    origin = glapp.camera.get_eye()
                    direction = (far_world[:3] - near_world[:3]).astype(np.float32)
                    n = np.linalg.norm(direction)
                    if n > 0:
                        direction /= n
                        # Find a layer with intersect_ray
                        target_layer = None
                        for ly in getattr(glapp.scene, '_layers', []):
                            if hasattr(ly, 'intersect_ray'):
                                target_layer = ly
                                break
                        if target_layer is not None:
                            hit = target_layer.intersect_ray(origin, direction)
                        # Plane y=0 fallback
                        if hit is None and abs(direction[1]) > 1e-6:
                            t = -origin[1] / direction[1]
                            if t > 0:
                                hit = origin + direction * t

                if hit is None:
                    # No useful ray: simple dolly (always apply)
                    glapp.camera.dolly(y_offset)
                    return

                # Move target towards hit and dolly smoothly
                zoom_speed = 0.18
                s = math.exp(-y_offset * zoom_speed)
                # clamp s to avoid crazy jumps
                s = float(np.clip(s, 0.05, 20.0))
                glapp.camera.distance = float(max(0.3, glapp.camera.distance * s))
                tgt = glapp.camera.target
                hitv = np.array(hit, dtype=np.float32)
                glapp.camera.target = tgt + (hitv - tgt) * (1.0 - s)

        try:
            import os
            os.environ.setdefault('MODERNGL_WINDOW', 'glfw')
            # Delegate to driver that mirrors the working example
            from .backend import run_glfw_app
            run_glfw_app(glapp)
        finally:
            sys.argv = argv_backup


def _update_imgui_display(win_w: Optional[int], win_h: Optional[int], wnd) -> None:
    try:
        io = imgui.get_io()
        if win_w is None or win_h is None:
            ww, wh = wnd.size
            win_w, win_h = int(ww), int(wh)
        fb_w, fb_h = wnd.buffer_size
        io.display_size = (float(max(1, win_w)), float(max(1, win_h)))
        sx = float(max(1, fb_w)) / float(max(1, win_w))
        sy = float(max(1, fb_h)) / float(max(1, win_h))
        io.display_fb_scale = (sx, sy)
    except Exception:
        pass
