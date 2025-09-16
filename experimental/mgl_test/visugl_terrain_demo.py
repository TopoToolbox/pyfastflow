#!/usr/bin/env python3
"""
Terrain demo using the new visuGL submodule

Demonstrates the lean, opinionated pyfastflow/visuGL/ architecture:
- create_3D_app() factory for easy setup
- Heightfield3D adapter with existing TerrainRenderer math
- DataHub for GPU texture management
- UI controls via ValueRef.subscribe()
- Real-time model updates possible

Usage: python experimental/mgl_test/visugl_terrain_demo.py
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Initialize Taichi CPU backend for compatibility
import taichi as ti
ti.init(arch=ti.cpu)

# Import PyFastFlow and the new visuGL system
import pyfastflow as pff
from pyfastflow import visuGL

def main():
    # Create 3D app with orbit camera
    app = visuGL.create_3D_app(title="visuGL Terrain Demo")

    # Create heightfield layer
    layer = visuGL.adapters.Heightfield3D(mesh_max_dim=512)
    app.scene.add(layer)

    # Generate initial heightmap with PyFastFlow
    def generate_heightmap():
        return pff.noise.perlin_noise(
            512, 512,
            frequency=1.0,
            octaves=6,
            persistence=0.6,
            seed=42
        )

    # Initialize data and wire up layer
    def init_callback(app):
        heightmap = generate_heightmap()
        app.data.tex2d("height", heightmap, "R32F")
        layer.use_hub("height")

    app.on_init(init_callback)

    # Create UI panel with controls
    panel = app.ui.add_panel("Terrain Controls", dock="right")

    # Z exaggeration slider
    z_scale = panel.slider("Z exaggeration", 0.5, 0.01, 5.0)
    z_scale.subscribe(layer.set_height_scale)

    # Sphere mode toggle
    sphere_mode = panel.checkbox("Sphere mode", False)
    sphere_mode.subscribe(layer.set_sphere_mode)

    # Light controls
    light_x = panel.slider("Light X", 0.5, -1.0, 1.0)
    light_y = panel.slider("Light Y", 1.0, 0.1, 2.0)
    light_z = panel.slider("Light Z", 0.5, -1.0, 1.0)

    def update_light():
        layer.set_light((light_x.value, light_y.value, light_z.value))

    light_x.subscribe(lambda _: update_light())
    light_y.subscribe(lambda _: update_light())
    light_z.subscribe(lambda _: update_light())

    # Regenerate heightmap button
    def regenerate():
        new_heightmap = generate_heightmap()
        app.data.update_tex("height", new_heightmap)

    panel.button("Regenerate", regenerate)

    # Enable docking for better UI layout
    app.ui.enable_docking(True)

    # Run the application
    app.run()

if __name__ == "__main__":
    main()