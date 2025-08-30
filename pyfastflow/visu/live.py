"""
Real-time 3D surface visualization for FastFlow Taichi.

Provides interactive 3D visualization of topographic surfaces, flow fields,
and other 2D scientific data using Taichi's built-in GUI system. Supports
real-time updates for dynamic simulations and interactive exploration.

Key Features:
- Real-time 3D mesh rendering with Taichi GUI
- Interactive camera controls and scaling
- Dynamic surface updates during simulation
- GPU-accelerated mesh generation
- Customizable lighting and materials

Author: B.G.
"""

import taichi as ti


@ti.data_oriented
class SurfaceViewer:
    """
    Interactive 3D surface viewer for topographic and scientific data.

    Creates a real-time 3D visualization window with interactive camera controls,
    adjustable scaling, and support for dynamic surface updates. Uses GPU-accelerated
    mesh generation for smooth performance with large datasets.

    Args:
        surface_data (np.ndarray): 2D array of surface heights (shape: height x width)
        x_scale (float): Horizontal scaling factor for x-axis (default: 2.0)
        y_scale (float): Vertical scaling factor for height (default: 0.3)
        z_scale (float): Horizontal scaling factor for z-axis (default: 2.0)
        window_size (tuple): Window dimensions in pixels (default: (1024, 768))

    Attributes:
        surface_data (np.ndarray): Current surface height data
        vertices (ti.Vector.field): 3D vertex positions for mesh
        indices (ti.field): Triangle indices for mesh connectivity
        camera (ti.ui.Camera): Interactive camera controller
        running (bool): Visualization loop control flag

    Controls:
        - Right mouse button + drag: Rotate camera
        - Mouse wheel: Zoom in/out
        - Spacebar: Reset camera to default position
        - GUI sliders: Adjust X, Y, Z scaling factors

    Example:
        import numpy as np
        import pyfastflow as pf

        # Create sample terrain
        terrain = np.random.rand(100, 100) * 1000

        # Start visualization
        viewer = pf.visu.SurfaceViewer(terrain)
        viewer.run()  # Interactive loop

        # Or integrate with simulation
        for step in range(1000):
            # Update terrain
            viewer.update_surface(new_terrain)
            if not viewer.render_frame():
                break

    Author: B.G.
    """

    def __init__(
        self,
        surface_data,
        x_scale=2.0,
        y_scale=0.3,
        z_scale=2.0,
        window_size=(1024, 768),
    ):
        self.surface_data = surface_data
        self.height, self.width = surface_data.shape
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale

        # Normalize surface data for consistent rendering
        self._normalize_surface()

        # Create mesh data structures
        self._setup_mesh()

        # Initialize Taichi GUI components
        self.window = ti.ui.Window("3D Surface Viewer", window_size)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.scene = self.window.get_scene()

        # Setup camera with default viewpoint
        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 1.0, 2.0)  # Elevated view
        self.camera.lookat(0.0, 0.0, 0.0)  # Look at center

        # Generate initial mesh geometry
        self._generate_mesh()
        self.needs_mesh_update = False

        self.running = True

    def _normalize_surface(self):
        """
        Normalize surface data to [0, 1] range for consistent rendering.

        Prevents rendering issues with extreme elevation values and ensures
        consistent visual scaling regardless of input data range.
        """
        height_min = self.surface_data.min()
        height_max = self.surface_data.max()
        height_range = height_max - height_min

        # Avoid division by zero for flat surfaces
        if height_range > 0:
            self.normalized_surface = (self.surface_data - height_min) / height_range
        else:
            self.normalized_surface = self.surface_data

    def _setup_mesh(self):
        """
        Initialize Taichi fields for mesh vertex and index data.

        Creates GPU-resident fields for storing 3D vertex positions and
        triangle connectivity indices. Mesh uses triangle primitives with
        two triangles per grid quad.
        """
        num_vertices = self.height * self.width
        num_faces = (self.height - 1) * (self.width - 1) * 2  # 2 triangles per quad

        # 3D vertex positions (x, y, z)
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

        # Triangle indices (3 vertices per triangle)
        self.indices = ti.field(dtype=ti.i32, shape=num_faces * 3)

    @ti.kernel
    def _generate_mesh_kernel(
        self,
        surface: ti.types.ndarray(),
        w: ti.i32,
        h: ti.i32,
        x_scale: ti.f32,
        y_scale: ti.f32,
        z_scale: ti.f32,
    ):
        """
        GPU kernel for generating 3D mesh from 2D surface data.

        Converts 2D height field into 3D triangular mesh with proper
        vertex positions and triangle connectivity. Uses parallel
        computation for efficient mesh generation.

        Args:
            surface: Normalized surface height data
            w, h: Surface dimensions (width, height)
            x_scale, y_scale, z_scale: Scaling factors for each axis
        """
        # Generate vertex positions in parallel
        for i, j in ti.ndrange(h, w):
            idx = i * w + j

            # Map grid coordinates to world space
            x = (j / (w - 1) - 0.5) * x_scale  # Center at origin
            z = (i / (h - 1) - 0.5) * z_scale  # Center at origin
            y = surface[i, j] * y_scale  # Scale height

            self.vertices[idx] = ti.Vector([x, y, z])

        # Generate triangle indices for each quad in parallel
        for i, j in ti.ndrange(h - 1, w - 1):
            quad_idx = i * (w - 1) + j

            # Vertex indices for current quad
            v0 = i * w + j  # Bottom-left
            v1 = i * w + (j + 1)  # Bottom-right
            v2 = (i + 1) * w + j  # Top-left
            v3 = (i + 1) * w + (j + 1)  # Top-right

            # First triangle (v0, v2, v1)
            self.indices[quad_idx * 6 + 0] = v0
            self.indices[quad_idx * 6 + 1] = v2
            self.indices[quad_idx * 6 + 2] = v1

            # Second triangle (v1, v2, v3)
            self.indices[quad_idx * 6 + 3] = v1
            self.indices[quad_idx * 6 + 4] = v2
            self.indices[quad_idx * 6 + 5] = v3

    def _generate_mesh(self):
        """
        Generate 3D mesh from current surface data and scaling parameters.

        Calls the GPU kernel to update vertex positions and triangle indices
        based on current surface data and user-specified scaling factors.
        """
        self._generate_mesh_kernel(
            self.normalized_surface,
            self.width,
            self.height,
            self.x_scale,
            self.y_scale,
            self.z_scale,
        )

    def update_surface(self, new_surface_data):
        """
        Update surface data for dynamic visualization.

        Replaces current surface with new height data and regenerates
        the 3D mesh. Useful for real-time visualization of changing
        terrain during simulations.

        Args:
            new_surface_data (np.ndarray): New 2D height data
                                          Must have same shape as original

        Raises:
            ValueError: If new surface shape doesn't match original

        Example:
            # During simulation loop
            for step in simulation_steps:
                new_terrain = simulate_step()
                viewer.update_surface(new_terrain)
                viewer.render_frame()
        """
        if new_surface_data.shape != (self.height, self.width):
            raise ValueError(
                f"Surface shape {new_surface_data.shape} must match "
                f"original {(self.height, self.width)}"
            )

        self.surface_data = new_surface_data.copy()
        self._normalize_surface()
        self._generate_mesh()  # Update mesh immediately
        self.needs_mesh_update = True

    def render_frame(self):
        """
        Render a single frame of the 3D visualization.

        Handles GUI controls, camera updates, mesh regeneration, lighting,
        and scene rendering. Call this method in your main loop for
        interactive visualization or integrate with simulation loops.

        Returns:
            bool: True if rendering successful, False if window closed

        Controls handled:
            - Scale sliders for real-time mesh adjustment
            - Camera controls (right-click drag, zoom)
            - Spacebar for camera reset

        Example:
            # Simple render loop
            while viewer.render_frame():
                time.sleep(0.016)  # ~60 FPS

            # Simulation integration
            for step in range(1000):
                new_data = simulate()
                viewer.update_surface(new_data)
                if not viewer.render_frame():
                    break
        """
        # Check if window was closed
        if not self.window.running:
            self.running = False
            return False

        # Update mesh if surface data changed
        if self.needs_mesh_update:
            self._generate_mesh()
            self.needs_mesh_update = False

        # Interactive scale controls in GUI panel
        with self.gui.sub_window("Scale Controls", 0.02, 0.02, 0.25, 0.25):
            old_x = self.x_scale
            old_y = self.y_scale
            old_z = self.z_scale

            # Sliders for real-time scaling adjustment
            self.x_scale = self.gui.slider_float("X Scale", self.x_scale, 0.1, 5.0)
            self.y_scale = self.gui.slider_float("Y Scale", self.y_scale, 0.01, 2.0)
            self.z_scale = self.gui.slider_float("Z Scale", self.z_scale, 0.1, 5.0)

            # Regenerate mesh if scaling changed
            if old_x != self.x_scale or old_y != self.y_scale or old_z != self.z_scale:
                self._generate_mesh()

        # Handle camera controls
        self.camera.track_user_inputs(
            self.window, movement_speed=0.05, hold_key=ti.ui.RMB
        )

        # Reset camera on spacebar
        if self.window.is_pressed(" "):
            self.camera.position(0.0, 1.0, 2.0)
            self.camera.lookat(0.0, 0.0, 0.0)

        # Setup scene lighting and render mesh
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.5, 0.5, 0.5))  # Soft ambient lighting
        self.scene.point_light(
            pos=(1.0, 2.0, 1.0), color=(1, 1, 1)
        )  # Directional light
        self.scene.mesh(
            self.vertices, self.indices, color=(0.7, 0.5, 0.3)
        )  # Terrain color

        # Display the rendered scene
        self.canvas.scene(self.scene)
        self.window.show()
        return True

    def run(self):
        """
        Start the interactive visualization loop.

        Runs continuous rendering until the window is closed or
        the viewer is stopped. Blocks execution until visualization
        ends. Use this for standalone visualization or call
        render_frame() manually for simulation integration.

        Example:
            viewer = SurfaceViewer(terrain_data)
            viewer.run()  # Interactive until window closed
        """
        while self.running and self.render_frame():
            pass

    def close(self):
        """
        Close the visualization window and stop rendering.

        Sets the running flag to False, which will cause the render
        loop to exit on the next frame. Safe to call from any thread.

        Example:
            # Close from simulation code
            if simulation_complete:
                viewer.close()
        """
        self.running = False
