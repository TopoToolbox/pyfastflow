"""
Lean, opinionated ModernGL viewer for PyFastFlow.

Public surface is intentionally tiny. Imports are lazy.
"""
from __future__ import annotations

__all__ = [
    # Factories
    "create_3D_app",
    "create_2D_app",
    # Core types
    "GLApp",
    "Scene",
    "Layer",
    "UI",
    "Panel",
    "DataHub",
    "OrbitCamera",
    # Adapters (namespaced)
    "adapters",
]


def __getattr__(name: str):
    if name in ("GLApp", "OrbitCamera"):
        from .app import GLApp, OrbitCamera  # lazy
        return {"GLApp": GLApp, "OrbitCamera": OrbitCamera}[name]
    if name in ("Scene", "Layer"):
        from .scene import Scene, Layer  # lazy
        return {"Scene": Scene, "Layer": Layer}[name]
    if name in ("UI", "Panel"):
        from .ui import UI, Panel  # lazy
        return {"UI": UI, "Panel": Panel}[name]
    if name == "DataHub":
        from .data import DataHub  # lazy
        return DataHub
    if name == "adapters":
        import importlib
        return importlib.import_module('.adapters', __name__)
    if name == "create_3D_app":
        def factory(title: str = "Displacement Viewer"):
            from .app import GLApp, OrbitCamera
            app = GLApp(title)
            app.camera = OrbitCamera()
            app.ui.enable_docking(True)
            return app
        return factory
    if name == "create_2D_app":
        def factory(title: str = "Array Viewer"):
            from .app import GLApp, OrbitCamera
            app = GLApp(title)
            app.camera = OrbitCamera()
            app.ui.enable_docking(True)
            # Optional: later swap to ortho if needed
            return app
        return factory
    raise AttributeError(name)
