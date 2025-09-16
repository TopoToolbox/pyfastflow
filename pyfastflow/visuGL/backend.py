from __future__ import annotations

import math
import numpy as np
import imgui
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer


def perspective(fovy_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_rad / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(1e-6, aspect)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def run_glfw_app(glapp) -> None:
    class Window(mglw.WindowConfig):
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
            # ImGui setup exactly like the working example
            imgui.create_context()
            self.imgui = ModernglWindowRenderer(self.wnd)
            io = imgui.get_io()
            io.config_flags |= int(getattr(imgui, "CONFIG_DOCKING_ENABLE", 0))
            # Inject ctx into data hub
            glapp.data._set_ctx(self.ctx)
            # Init + scene setup
            for fn in glapp._init_cbs:
                try:
                    fn(glapp)
                except Exception:
                    pass
            glapp.scene._setup(self.ctx, glapp.data)
            # Event state
            self._evt_counts = {"pos": 0, "drag": 0, "press": 0, "release": 0, "scroll": 0}
            self._last_event_pos = (float('nan'), float('nan'))
            self._last_mouse_pos = (float('nan'), float('nan'))
            self._last_evt_dxdy = (0.0, 0.0)
            self.relative_mouse = False
            self._last = 0.0
            self._last_proj = None
            self._last_view = None
            self._wheel_accum = 0.0
            # Initialize last raw cursor to avoid first-frame jump
            try:
                raw = self._get_glfw_window()
                if raw is not None:
                    import glfw  # type: ignore
                    cx, cy = glfw.get_cursor_pos(raw)
                    if all(np.isfinite(v) for v in (cx, cy)):
                        self._last_mouse_pos = (float(cx), float(cy))
            except Exception:
                pass
            # Install GLFW scroll callback wrapper to capture wheel reliably
            try:
                raw = self._get_glfw_window()
                if raw is not None:
                    import glfw  # type: ignore
                    try:
                        prev_cb = glfw.get_scroll_callback(raw)
                    except Exception:
                        prev_cb = None
                    def _scroll_cb(win, xoff, yoff):
                        try:
                            if prev_cb is not None:
                                prev_cb(win, xoff, yoff)
                        except Exception:
                            pass
                        try:
                            self._wheel_accum += float(yoff)
                        except Exception:
                            pass
                    glfw.set_scroll_callback(raw, _scroll_cb)
            except Exception:
                pass

        # IO helpers ------------------------------------------------------
        def _update_imgui_display(self, win_w: int, win_h: int, fb_w: int | None = None, fb_h: int | None = None):
            try:
                io = imgui.get_io()
                io.display_size = (float(max(1, win_w)), float(max(1, win_h)))
                if fb_w is None or fb_h is None:
                    fb_w, fb_h = self.wnd.buffer_size
                sx = float(max(1, fb_w)) / float(max(1, win_w))
                sy = float(max(1, fb_h)) / float(max(1, win_h))
                io.display_fb_scale = (sx, sy)
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

        # Events ----------------------------------------------------------
        def key_event(self, key, action, modifiers):
            try:
                self.imgui.key_event(key, action, modifiers)
            except Exception:
                pass
            if action == self.wnd.keys.ACTION_PRESS:
                if key == self.wnd.keys.ESCAPE:
                    self.wnd.close()
                if key == self.wnd.keys.F11:
                    try:
                        self.wnd.toggle_fullscreen()
                    except Exception:
                        pass

        def mouse_press_event(self, x: int, y: int, button: int):
            try:
                self.imgui.mouse_press_event(x, y, button)
            except Exception:
                pass
            self._evt_counts['press'] += 1
            if all(np.isfinite(v) for v in (x, y)):
                self._last_event_pos = (float(x), float(y))

        def mouse_release_event(self, x: int, y: int, button: int):
            try:
                self.imgui.mouse_release_event(x, y, button)
            except Exception:
                pass
            self._evt_counts['release'] += 1
            if all(np.isfinite(v) for v in (x, y)):
                self._last_event_pos = (float(x), float(y))

        def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
            try:
                self.imgui.mouse_drag_event(x, y, dx, dy)
            except Exception:
                pass
            if all(np.isfinite(v) for v in (x, y)):
                self._last_event_pos = (float(x), float(y))
            self._evt_counts['drag'] += 1
            # Camera movement handled per-frame from raw cursor for smoothness

        def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
            try:
                self.imgui.mouse_position_event(x, y, dx, dy)
            except Exception:
                pass
            if all(np.isfinite(v) for v in (x, y)):
                self._last_event_pos = (float(x), float(y))
            self._evt_counts['pos'] += 1
            # Camera movement handled per-frame from raw cursor for smoothness

        def mouse_scroll_event(self, x_offset: float, y_offset: float):
            try:
                self.imgui.mouse_scroll_event(x_offset, y_offset)
            except Exception:
                pass
            self._evt_counts['scroll'] += 1
            self._apply_zoom_to_mouse(y_offset)

        # Some backends use this event name
        def scroll_event(self, x_offset: float, y_offset: float):
            return self.mouse_scroll_event(x_offset, y_offset)

        def unicode_char_entered(self, char: str):
            try:
                self.imgui.unicode_char_entered(char)
            except Exception:
                pass

        # Resize events ---------------------------------------------------
        def resize(self, width: int, height: int):
            self.ctx.viewport = (0, 0, max(1, width), max(1, height))
            try:
                self.imgui.resize(width, height)
                self._update_imgui_display(width, height)
            except Exception:
                pass

        def buffer_size_event(self, width: int, height: int):
            try:
                self.ctx.viewport = (0, 0, max(1, width), max(1, height))
                try:
                    win_w, win_h = self.wnd.size
                except Exception:
                    win_w, win_h = width, height
                self._update_imgui_display(win_w, win_h, width, height)
            except Exception:
                pass

        # Helpers ---------------------------------------------------------
        def _apply_camera_drag(self, dx: float, dy: float):
            ms = getattr(self.wnd, 'mouse_states', None)
            # Also read raw GLFW buttons in case mouse_states isn't updated
            raw_left = raw_right = raw_middle = False
            try:
                raw = self._get_glfw_window()
                if raw is not None:
                    import glfw  # type: ignore
                    raw_left = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                    raw_right = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                    raw_middle = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            except Exception:
                pass
            if not ms and not (raw_left or raw_right or raw_middle):
                return
            sensitivity = 0.5
            if (getattr(ms, 'left', False) if ms else False) or raw_left:
                glapp.camera.yaw += dx * sensitivity
                glapp.camera.pitch = float(np.clip(glapp.camera.pitch - dy * sensitivity, -89.0, 89.0))
            elif (getattr(ms, 'right', False) if ms else False) or (getattr(ms, 'middle', False) if ms else False) or raw_right or raw_middle:
                pr = math.radians(glapp.camera.pitch)
                yr = math.radians(glapp.camera.yaw)
                right = np.array([-math.sin(yr), 0.0, math.cos(yr)], dtype=np.float32)
                forward = np.array([math.cos(pr) * math.cos(yr), math.sin(pr), math.cos(pr) * math.sin(yr)], dtype=np.float32)
                right /= max(1e-6, np.linalg.norm(right))
                forward /= max(1e-6, np.linalg.norm(forward))
                up = np.cross(right, forward)
                up /= max(1e-6, np.linalg.norm(up))
                pan_speed = 0.01 * glapp.camera.distance
                glapp.camera.target = glapp.camera.target + right * dx * pan_speed + up * (-dy) * pan_speed

        def _raw_cursor_delta_window(self):
            dx = dy = 0.0
            try:
                raw_win = self._get_glfw_window()
                if raw_win is None:
                    return 0.0, 0.0
                import glfw  # type: ignore
                rcx, rcy = glfw.get_cursor_pos(raw_win)
                if all(np.isfinite(v) for v in (rcx, rcy)):
                    # Compute raw deltas
                    if not all(np.isfinite(v) for v in self._last_mouse_pos):
                        self._last_mouse_pos = (float(rcx), float(rcy))
                    dx_raw = float(rcx - self._last_mouse_pos[0])
                    dy_raw = float(rcy - self._last_mouse_pos[1])
                    self._last_mouse_pos = (float(rcx), float(rcy))
                    # Map to window-space deltas if fb != win
                    try:
                        ww, wh = self.wnd.size
                        fbw, fbh = self.wnd.buffer_size
                        sx = float(fbw) / max(1.0, float(ww))
                        sy = float(fbh) / max(1.0, float(wh))
                    except Exception:
                        sx = sy = 1.0
                    if sx > 0 and sy > 0:
                        dx = dx_raw / sx
                        dy = dy_raw / sy
                    else:
                        dx, dy = dx_raw, dy_raw
            except Exception:
                pass
            return dx, dy

        def _apply_zoom_to_mouse(self, y_offset: float):
            io = imgui.get_io()
            try:
                win_w, win_h = io.display_size
                win_w = float(win_w)
                win_h = float(win_h)
            except Exception:
                try:
                    ww, wh = self.wnd.size
                    win_w, win_h = float(ww), float(wh)
                except Exception:
                    win_w, win_h = float(self.window_size[0]), float(self.window_size[1])

            mp = getattr(io, 'mouse_pos', (-1.0, -1.0))
            mx, my = float(mp[0]), float(mp[1])
            if not (np.isfinite(mx) and np.isfinite(my)):
                try:
                    mx2, my2 = self.wnd.mouse_position
                    if np.isfinite(mx2) and np.isfinite(my2):
                        mx, my = float(mx2), float(my2)
                except Exception:
                    pass
            if not (np.isfinite(mx) and np.isfinite(my)):
                mx, my = win_w * 0.5, win_h * 0.5

            proj = self._last_proj
            view = self._last_view
            if proj is None or view is None:
                aspect = max(1.0, win_w / max(1.0, win_h))
                proj = perspective(math.radians(60.0), aspect, 0.1, 100.0)
                view = glapp.camera.get_view_matrix()

            vp = proj @ view
            try:
                inv_vp = np.linalg.inv(vp)
            except Exception:
                inv_vp = None

            hit = None
            if inv_vp is not None:
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
                    target_layer = None
                    for ly in getattr(glapp.scene, '_layers', []):
                        if hasattr(ly, 'intersect_ray'):
                            target_layer = ly
                            break
                    if target_layer is not None:
                        hit = target_layer.intersect_ray(origin, direction)
                    if hit is None and abs(direction[1]) > 1e-6:
                        t = -origin[1] / direction[1]
                        if t > 0:
                            hit = origin + direction * t

            zoom_speed = 0.18
            s = math.exp(-y_offset * zoom_speed)
            s = max(0.1, min(10.0, s))
            if hit is not None:
                glapp.camera.target = glapp.camera.target + (1.0 - s) * (hit - glapp.camera.target)
            glapp.camera.distance = max(0.5, glapp.camera.distance * s)

        # Render loop -----------------------------------------------------
        def on_render(self, time: float, frame_time: float):
            fb_w, fb_h = self.wnd.buffer_size
            self.ctx.viewport = (0, 0, fb_w, fb_h)
            try:
                self.ctx.clear(0.05, 0.05, 0.08, 1.0, depth=1.0)
            except TypeError:
                self.ctx.clear(0.05, 0.05, 0.08, 1.0)
            # Apply camera movement first for immediate responsiveness
            try:
                io2 = self.imgui.io
                if not io2.want_capture_mouse:
                    # Determine button states (raw + mouse_states)
                    raw_left = raw_right = raw_middle = False
                    try:
                        raw = self._get_glfw_window()
                        if raw is not None:
                            import glfw  # type: ignore
                            raw_left = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
                            raw_right = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
                            raw_middle = glfw.get_mouse_button(raw, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
                    except Exception:
                        pass
                    ms = getattr(self.wnd, 'mouse_states', None)
                    left_on = (getattr(ms, 'left', False) if ms else False) or raw_left
                    right_on = (getattr(ms, 'right', False) if ms else False) or raw_right or raw_middle
                    if left_on or right_on:
                        dx, dy = self._raw_cursor_delta_window()
                        self._apply_camera_drag(dx, dy)
            except Exception:
                pass
            proj = perspective(math.radians(60.0), max(1.0, fb_w / max(1.0, fb_h)), 0.1, 100.0)
            view = glapp.camera.get_view_matrix()
            self._last_proj = proj
            self._last_view = view
            # Draw scene
            glapp.scene.draw_all(self.ctx, glapp.camera)

            # ImGui IO and new frame
            try:
                if (self._evt_counts['pos'] + self._evt_counts['drag']) == 0 and 'glfw' in type(self.wnd).__module__:
                    raw_win = self._get_glfw_window()
                    if raw_win is not None:
                        import glfw  # type: ignore
                        cx, cy = glfw.get_cursor_pos(raw_win)
                        if all(np.isfinite(v) for v in (cx, cy)):
                            self._last_event_pos = (float(cx), float(cy))
                            self._last_mouse_pos = (float(cx), float(cy))
            except Exception:
                pass
            try:
                win_w, win_h = self.wnd.size
                self._update_imgui_display(win_w, win_h, fb_w, fb_h)
                self.imgui.process_inputs()
            except Exception:
                pass
            io_fix = imgui.get_io()
            # Force-feed mouse position with proper coordinate space conversion
            try:
                # Get raw cursor in the most reliable space available
                raw_win = self._get_glfw_window()
                cx = cy = None
                if raw_win is not None:
                    import glfw  # type: ignore
                    _cx, _cy = glfw.get_cursor_pos(raw_win)
                    if all(np.isfinite(v) for v in (_cx, _cy)):
                        cx, cy = float(_cx), float(_cy)
                if cx is None or cy is None:
                    mx, my = self.wnd.mouse_position
                    if np.isfinite(mx) and np.isfinite(my):
                        cx, cy = float(mx), float(my)
                if cx is not None and cy is not None:
                    # If cursor appears to be in framebuffer coords, map to window coords
                    ww, wh = self.wnd.size
                    fbw, fbh = self.wnd.buffer_size
                    if (cx > ww + 0.5) or (cy > wh + 0.5) or (abs(fbw - ww) > 1 or abs(fbh - wh) > 1):
                        # Convert: window_coord = cursor / fb * win
                        cx = cx / max(1.0, float(fbw)) * float(ww)
                        cy = cy / max(1.0, float(fbh)) * float(wh)
                    io_fix.mouse_pos = (cx, cy)
                    self._last_event_pos = (cx, cy)
            except Exception:
                # As a last resort, keep last_event_pos
                if not all(np.isfinite(v) for v in io_fix.mouse_pos):
                    io_fix.mouse_pos = self._last_event_pos
            try:
                ms = getattr(self.wnd, 'mouse_states', None)
                if ms:
                    io_fix.mouse_down[0] = bool(ms.left)
                    io_fix.mouse_down[1] = bool(ms.right)
                    io_fix.mouse_down[2] = bool(ms.middle)
            except Exception:
                pass
            imgui.new_frame()
            # Fallback wheel: some backends don't fire scroll events reliably
            try:
                wheel = float(getattr(io_fix, 'mouse_wheel', 0.0)) + float(self._wheel_accum)
                if abs(wheel) > 1e-6:
                    self._apply_zoom_to_mouse(wheel)
                    self._wheel_accum = 0.0
            except Exception:
                pass
            # Build user UI
            for fn in glapp._gui_cbs:
                try:
                    fn()
                except Exception:
                    pass
            glapp.ui._draw_all_panels(imgui)

            # Debug overlay to inspect IO and events
            try:
                imgui.set_next_window_bg_alpha(0.85)
                imgui.set_next_window_position(12.0, 12.0)
                flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE |
                         imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS)
                imgui.begin("IO-Debug", True, flags)
                io_dbg = imgui.get_io()
                imgui.text(f"want_capture_mouse={bool(io_dbg.want_capture_mouse)}")
                imgui.text(f"display_size={tuple(map(int, io_dbg.display_size))}")
                imgui.text(f"display_fb_scale={io_dbg.display_fb_scale}")
                try:
                    ww, wh = self.wnd.size
                    fbw, fbh = self.wnd.buffer_size
                    imgui.text(f"wnd.size={int(ww)},{int(wh)} fb={int(fbw)},{int(fbh)}")
                except Exception:
                    pass
                try:
                    mp = tuple(io_dbg.mouse_pos)
                    imgui.text(f"io.mouse_pos={int(mp[0])},{int(mp[1])}")
                except Exception:
                    pass
                try:
                    raw_win = self._get_glfw_window()
                    if raw_win is not None:
                        import glfw  # type: ignore
                        cx, cy = glfw.get_cursor_pos(raw_win)
                        imgui.text(f"glfw.cursor={int(cx)},{int(cy)}")
                except Exception:
                    pass
                imgui.text(f"mouse_down={tuple(int(x) for x in io_dbg.mouse_down[:3])}")
                imgui.text(f"evt pos/drag/press/release/scroll={self._evt_counts['pos']}/{self._evt_counts['drag']}/{self._evt_counts['press']}/{self._evt_counts['release']}/{self._evt_counts['scroll']}")
                imgui.end()
            except Exception:
                pass
            imgui.render()
            self.imgui.render(imgui.get_draw_data())

            # (removed end-of-frame camera application to reduce latency)

        # Compatibility with older moderngl-window
        def render(self, t: float, dt: float):
            return self.on_render(t, dt)

    mglw.run_window_config(Window)
