import taichi as ti
import numpy as np
import threading
import time

@ti.data_oriented
class SurfaceViewer:
    def __init__(self, surface_data, x_scale=2.0, y_scale=0.3, z_scale=2.0, window_size=(1024, 768)):
        ti.init(arch=ti.gpu)
        
        self.surface_data = surface_data
        self.height, self.width = surface_data.shape
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale
        
        # Normalize surface data
        self._normalize_surface()
        
        # Create mesh data structures
        self._setup_mesh()
        
        # Create GUI
        self.window = ti.ui.Window("3D Surface Viewer", window_size)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()
        self.camera.position(0.0, 1.0, 2.0)
        self.camera.lookat(0.0, 0.0, 0.0)
        
        # Generate initial mesh
        self._generate_mesh()
        self.needs_mesh_update = False
        
        self.running = True
        
    def _normalize_surface(self):
        height_min = self.surface_data.min()
        height_max = self.surface_data.max()
        height_range = height_max - height_min
        self.normalized_surface = (self.surface_data - height_min) / height_range if height_range > 0 else self.surface_data
        
    def _setup_mesh(self):
        num_vertices = self.height * self.width
        num_faces = (self.height - 1) * (self.width - 1) * 2
        
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
        self.indices = ti.field(dtype=ti.i32, shape=num_faces * 3)
        
    @ti.kernel
    def _generate_mesh_kernel(self, surface: ti.types.ndarray(), w: ti.i32, h: ti.i32, 
                              x_scale: ti.f32, y_scale: ti.f32, z_scale: ti.f32):
        # Generate vertices
        for i, j in ti.ndrange(h, w):
            idx = i * w + j
            x = (j / (w - 1) - 0.5) * x_scale
            z = (i / (h - 1) - 0.5) * z_scale
            y = surface[i, j] * y_scale
            self.vertices[idx] = ti.Vector([x, y, z])
        
        # Generate triangle indices
        for i, j in ti.ndrange(h - 1, w - 1):
            quad_idx = i * (w - 1) + j
            
            v0 = i * w + j
            v1 = i * w + (j + 1)
            v2 = (i + 1) * w + j
            v3 = (i + 1) * w + (j + 1)
            
            self.indices[quad_idx * 6 + 0] = v0
            self.indices[quad_idx * 6 + 1] = v2
            self.indices[quad_idx * 6 + 2] = v1
            
            self.indices[quad_idx * 6 + 3] = v1
            self.indices[quad_idx * 6 + 4] = v2
            self.indices[quad_idx * 6 + 5] = v3
            
    def _generate_mesh(self):
        self._generate_mesh_kernel(self.normalized_surface, self.width, self.height, 
                                   self.x_scale, self.y_scale, self.z_scale)
    
    def update_surface(self, new_surface_data):
        """Update the surface data - must be same dimensions"""
        if new_surface_data.shape != (self.height, self.width):
            raise ValueError(f"Surface shape {new_surface_data.shape} must match original {(self.height, self.width)}")
        
        self.surface_data = new_surface_data.copy()
        self._normalize_surface()
        self._generate_mesh()  # Update immediately
        self.needs_mesh_update = True

    
    def render_frame(self):
        """Render one frame - call this in your main loop"""
        if not self.window.running:
            self.running = False
            return False
        
        # Update mesh if needed (only in render thread)
        if self.needs_mesh_update:
            self._generate_mesh()
            self.needs_mesh_update = False
            
        # GUI controls
        with self.gui.sub_window("Scale Controls", 0.02, 0.02, 0.25, 0.25):
            old_x = self.x_scale
            old_y = self.y_scale
            old_z = self.z_scale
            
            self.x_scale = self.gui.slider_float("X Scale", self.x_scale, 0.1, 5.0)
            self.y_scale = self.gui.slider_float("Y Scale", self.y_scale, 0.01, 2.0)
            self.z_scale = self.gui.slider_float("Z Scale", self.z_scale, 0.1, 5.0)
            
            if old_x != self.x_scale or old_y != self.y_scale or old_z != self.z_scale:
                self._generate_mesh()
        
        # Camera controls
        self.camera.track_user_inputs(self.window, movement_speed=0.05, hold_key=ti.ui.RMB)
        
        if self.window.is_pressed(' '):
            self.camera.position(0.0, 1.0, 2.0)
            self.camera.lookat(0.0, 0.0, 0.0)
        
        # Render
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.point_light(pos=(1.0, 2.0, 1.0), color=(1, 1, 1))
        self.scene.mesh(self.vertices, self.indices, color=(0.7, 0.5, 0.3))
        
        self.canvas.scene(self.scene)
        self.window.show()
        return True
    
    def run(self):
        """Run the visualization loop"""
        while self.running and self.render_frame():
            pass
    
    def close(self):
        """Close the viewer"""
        self.running = False


# Example usage:
if __name__ == "__main__":
    # Load initial surface
    surface = np.load('your_surface.npy')
    
    # Create viewer
    viewer = SurfaceViewer(surface)
    
    # Main loop - render in main thread only
    for i in range(1000):
        # Your computation here
        new_surface = surface + np.random.rand(*surface.shape)*100
        viewer.update_surface(new_surface)
        
        # Render frame
        if not viewer.render_frame():
            break
        
        # Small delay to control update rate
        time.sleep(0.01)