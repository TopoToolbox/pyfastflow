#!/usr/bin/env python3
"""
Simple test of the visuGL system
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Initialize Taichi CPU backend for compatibility
import taichi as ti
ti.init(arch=ti.cpu)

# Import the new visuGL system
from pyfastflow import visuGL

def main():
    try:
        # Create 3D app
        app = visuGL.create_3D_app(title="visuGL Simple Test")
        print("App created successfully")

        # Create simple heightfield
        layer = visuGL.adapters.Heightfield3D(mesh_max_dim=128)
        app.scene.add(layer)
        print("Layer added successfully")

        # Create simple heightmap
        def init_callback(app_instance):
            print("Init callback called")
            # Simple test heightmap
            size = 64
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            heightmap = np.exp(-(X*X + Y*Y)).astype(np.float32)

            app_instance.data.tex2d("height", heightmap, "R32F")
            layer.use_hub("height")
            print("Heightmap created and uploaded")

        app.on_init(init_callback)

        print("Starting app...")
        app.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()