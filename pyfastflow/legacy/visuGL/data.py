from __future__ import annotations

import numpy as np
import moderngl
from typing import Dict, Any, Optional, Tuple


class DataHub:
    """Minimal GPU data hub. Owns uploads only.

    Use after GL is alive. The app will inject `ctx`.
    """

    def __init__(self):
        self._ctx = None
        self._tex2d: Dict[str, Dict[str, Any]] = {}

    # App-internal
    def _set_ctx(self, ctx):
        self._ctx = ctx

    # Public API -----------------------------------------------------------
    def tex2d(self, name: str, np_array: np.ndarray, fmt: str = "R32F", filter: str = "linear") -> Any:
        assert self._ctx is not None, "GL context not initialized yet"
        arr = np.asarray(np_array)
        assert arr.flags.c_contiguous, "np_array must be C-contiguous"
        assert arr.dtype in (np.float32, np.uint8), "dtype must be float32 or uint8"
        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
        elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
            h, w, c = arr.shape
            arr = arr.reshape(h, w, c)
        else:
            raise AssertionError("shape must be (H,W) or (H,W,C) with C in {1,2,3,4}")

        # Determine internal format
        dtype = arr.dtype
        channels = c
        if dtype == np.float32:
            internal = {1: 'R32F', 2: 'RG32F', 3: 'RGB32F', 4: 'RGBA32F'}[channels]
        else:
            internal = {1: 'R8', 2: 'RG8', 3: 'RGB8', 4: 'RGBA8'}[channels]
        if fmt:
            internal = fmt  # stored for reference only

        # Create texture (ModernGL infers internal format from dtype/components)
        tex = self._ctx.texture((w, h), channels, data=arr.tobytes(), dtype='f4' if dtype == np.float32 else 'u1')
        if filter == "linear":
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False

        info = {
            "tex": tex,
            "shape": (h, w),
            "dtype": dtype,
            "channels": channels,
            "fmt": internal,
            "cpu": arr,
        }
        self._tex2d[name] = info
        return tex

    def update_tex(self, name: str, np_array: np.ndarray) -> None:
        assert self._ctx is not None, "GL context not initialized yet"
        assert name in self._tex2d, f"unknown texture: {name}"
        arr = np.asarray(np_array)
        assert arr.flags.c_contiguous, "np_array must be C-contiguous"
        if arr.ndim == 2:
            h, w = arr.shape
            c = 1
        elif arr.ndim == 3 and arr.shape[2] in (1, 2, 3, 4):
            h, w, c = arr.shape
        else:
            raise AssertionError("shape must be (H,W) or (H,W,C) with C in {1,2,3,4}")

        info = self._tex2d[name]
        same_shape = (h, w) == tuple(info["shape"]) and c == int(info["channels"]) and arr.dtype == info["dtype"]
        if same_shape:
            # Update subimage fast path
            tex = info["tex"]
            tex.write(arr.tobytes())
            info["cpu"] = arr
        else:
            # Reallocate
            tex = info["tex"]
            try:
                tex.release()
            except Exception:
                pass
            self.tex2d(name, arr, fmt=info["fmt"])

    # Convenience getter
    def get_tex(self, name: str) -> Optional[Any]:
        ent = self._tex2d.get(name)
        return ent["tex"] if ent else None

    def get_cpu(self, name: str):
        ent = self._tex2d.get(name)
        return ent.get("cpu") if ent else None
