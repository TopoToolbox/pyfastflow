#!/usr/bin/env python3
"""
Modern OpenGL 3D Terrain Demo with PyFastFlow Perlin Noise

A comprehensive demonstration of ModernGL + GLFW + PyImgui for 3D terrain rendering.
Uses PyFastFlow's GPU-accelerated Perlin noise to generate realistic heightmaps
and renders them as a 3D mesh with normal-based lighting.

Features:
- Real-time 3D mesh rendering with ModernGL
- GPU-accelerated Perlin noise via PyFastFlow
- Interactive camera controls (rotate, pan, zoom)
- ImGui parameter interface
- Normal-based lighting with configurable light direction

Controls:
- Left Mouse: Rotate camera around center
- Right Mouse: Pan center of rotation
- Mouse Wheel: Zoom in/out
- ImGui Panel: Adjust noise and rendering parameters

Author: B.G.
"""

import numpy as np
import moderngl
import glfw
import imgui
import math
from typing import Tuple, Optional

# Import PyFastFlow for Perlin noise generation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import taichi as ti
import pyfastflow as pff

# Initialize Taichi on CUDA explicitly as requested
ti.init(arch=ti.gpu)


class Camera:
    """RTS-style camera with orbit controls around a target point."""
    
    def __init__(self, target=(0.0, 0.0, 0.0), distance=5.0, pitch=30.0, yaw=0.0):
        self.target = np.array(target, dtype=np.float32)
        self.distance = distance
        self.pitch = pitch  # Vertical rotation (degrees)
        self.yaw = yaw      # Horizontal rotation (degrees)


        # Mouse control state
        self.last_mouse = (0, 0)
        self.mouse_captured = None
        
    def get_view_matrix(self):
        """Calculate view matrix from current camera parameters."""
        # Convert angles to radians
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        
        # Calculate camera position relative to target
        x = self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        z = self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.distance * math.sin(pitch_rad)
        
        eye = self.target + np.array([x, y, z])
        up = np.array([0.0, 1.0, 0.0])
        
        return self._look_at(eye, self.target, up)

    def get_eye(self) -> np.ndarray:
        """Return camera eye position in world space."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        x = self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        z = self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.distance * math.sin(pitch_rad)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def get_forward(self) -> np.ndarray:
        """Return forward/look direction (unit) in world space."""
        pitch_rad = math.radians(self.pitch)
        yaw_rad = math.radians(self.yaw)
        f = np.array([
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad),
        ], dtype=np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 0 else f
    
    def _look_at(self, eye, target, up):
        """Create a look-at view matrix."""
        f = target - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        result = np.eye(4, dtype=np.float32)
        result[0, 0:3] = s
        result[1, 0:3] = u
        result[2, 0:3] = -f
        result[0, 3] = -np.dot(s, eye)
        result[1, 3] = -np.dot(u, eye)
        result[2, 3] = np.dot(f, eye)
        
        return result
    
    def handle_mouse(self, window, button, action, mods):
        """Handle mouse button events for camera control."""
        if action == glfw.PRESS:
            self.mouse_captured = button
            self.last_mouse = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self.mouse_captured = None
    
    def handle_cursor(self, window, xpos, ypos):
        """Handle mouse movement for camera rotation and panning."""
        if self.mouse_captured is None:
            return
            
        dx = xpos - self.last_mouse[0]
        dy = ypos - self.last_mouse[1]
        self.last_mouse = (xpos, ypos)
        
        sensitivity = 0.5
        
        if self.mouse_captured == glfw.MOUSE_BUTTON_LEFT:
            # Rotate camera around target
            self.yaw += dx * sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch - dy * sensitivity))

        elif self.mouse_captured == glfw.MOUSE_BUTTON_RIGHT:
            # Pan target point relative to current camera view
            pitch_rad = math.radians(self.pitch)
            yaw_rad = math.radians(self.yaw)
            
            # Camera's right vector (perpendicular to look direction)
            right = np.array([-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)])
            # Camera's up vector (perpendicular to both look and right)
            forward = np.array([math.cos(pitch_rad) * math.cos(yaw_rad), 
                               math.sin(pitch_rad),
                               math.cos(pitch_rad) * math.sin(yaw_rad)])
            # Normalize vectors to ensure consistent pan speed
            right = right / np.linalg.norm(right)
            forward = forward / np.linalg.norm(forward)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            pan_speed = 0.01 * self.distance
            # Intuitive panning: drag right -> move right, drag up -> move up
            self.target += right * dx * pan_speed
            self.target += up * (-dy) * pan_speed
    
    def handle_scroll(self, window, xoffset, yoffset):
        """Handle mouse wheel for zooming."""
        zoom_speed = 0.1
        self.distance = max(0.5, self.distance - yoffset * zoom_speed * self.distance)


class TerrainRenderer:
    """3D terrain renderer using ModernGL with heightmap displacement and lighting."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.mesh_size = 4.0  # Physical size of the mesh in world units
        
        # Noise generation parameters
        self.noise_params = {
            'frequency': 1.,
            'octaves': 6,
            'persistence': 0.6,
            'amplitude': 1.0,
            'seed': 42
        }
        
        # Rendering parameters
        self.light_direction = np.array([0.5, 1.0, 0.5], dtype=np.float32)
        self.light_direction /= np.linalg.norm(self.light_direction)
        self.height_scale = 0.5  # Used in shader and CPU intersection
        self.heightmap_cpu: Optional[np.ndarray] = None

        self.mesh_dim = 1024
        
        # Initialize OpenGL context and shaders
        self._init_gl()
        self._create_shaders()
        self._generate_mesh()
        self._update_heightmap()

    def _init_gl(self):
        """Initialize OpenGL context via ModernGL."""
        self.ctx = moderngl.create_context()
        # Depth test for 3D terrain
        self.ctx.enable(moderngl.DEPTH_TEST)
        # Culling off so we always see the mesh regardless of windings
        self.ctx.disable(moderngl.CULL_FACE)

    def _create_shaders(self):
        """Create vertex and fragment shaders for terrain rendering."""

        vertex_shader = """
        #version 330 core

        in vec3 in_position;
        in vec2 in_texcoord;

        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform sampler2D u_heightmap;
        uniform float u_height_scale;

        out vec3 v_world_pos;
        out vec3 v_normal;
        out float v_height;

        void main() {
            float height = texture(u_heightmap, in_texcoord).r * u_height_scale;
            vec3 displaced_pos = in_position + vec3(0.0, height, 0.0);
            v_world_pos = (u_model * vec4(displaced_pos, 1.0)).xyz;
            v_height = height;
            vec2 texel_size = 1.0 / textureSize(u_heightmap, 0);
            float h_right = texture(u_heightmap, in_texcoord + vec2(texel_size.x, 0.0)).r;
            float h_left = texture(u_heightmap, in_texcoord - vec2(texel_size.x, 0.0)).r;
            float h_up = texture(u_heightmap, in_texcoord + vec2(0.0, texel_size.y)).r;
            float h_down = texture(u_heightmap, in_texcoord - vec2(0.0, texel_size.y)).r;
            vec3 normal = normalize(vec3(
                (h_left - h_right) * u_height_scale,
                2.0 * texel_size.x,
                (h_down - h_up) * u_height_scale
            ));
            v_normal = normalize((u_model * vec4(normal, 0.0)).xyz);
            gl_Position = u_projection * u_view * vec4(v_world_pos, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core

        in vec3 v_world_pos;
        in vec3 v_normal;
        in float v_height;

        uniform vec3 u_light_direction;
        uniform float u_min_height;
        uniform float u_max_height;

        out vec4 frag_color;

        vec3 height_to_color(float normalized_height) {
            vec3 low_color = vec3(0.2, 0.4, 0.1);
            vec3 mid_color = vec3(0.6, 0.5, 0.3);
            vec3 high_color = vec3(0.9, 0.9, 0.8);
            if (normalized_height < 0.5) {
                return mix(low_color, mid_color, normalized_height * 2.0);
            } else {
                return mix(mid_color, high_color, (normalized_height - 0.5) * 2.0);
            }
        }

        void main() {
            float normalized_height = (v_height - u_min_height) / (u_max_height - u_min_height);
            normalized_height = clamp(normalized_height, 0.0, 1.0);
            vec3 base_color = height_to_color(normalized_height);
            float n_dot_l = max(0.0, dot(normalize(v_normal), u_light_direction));
            float ambient = 0.3;
            float lighting = ambient + (1.0 - ambient) * n_dot_l;
            frag_color = vec4(base_color * lighting, 1.0);
        }
        """

        self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

    def _generate_mesh(self):
        """Generate a 2D mesh with self.mesh_dim nodes on the longest dimension."""
        aspect_ratio = self.width / self.height
        if self.width >= self.height:
            nx = self.mesh_dim
            ny = int(self.mesh_dim / aspect_ratio)
        else:
            nx = int(self.mesh_dim * aspect_ratio)
            ny = self.mesh_dim
        self.mesh_nx, self.mesh_ny = nx, ny
        vertices = []
        texcoords = []
        for j in range(ny):
            for i in range(nx):
                float_i = i / (nx - 1)
                float_j = j / (ny - 1)
                x = (float_i - 0.5) * self.mesh_size
                z = (float_j - 0.5) * self.mesh_size
                y = 0.0
                vertices.extend([x, y, z])
                texcoords.extend([float_i, float_j])
        indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                tl = j * nx + i
                tr = j * nx + i + 1
                bl = (j + 1) * nx + i
                br = (j + 1) * nx + i + 1
                indices.extend([tl, bl, tr])
                indices.extend([tr, bl, br])
        vertices = np.array(vertices, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        self.vbo_vertices = self.ctx.buffer(vertices)
        self.vbo_texcoords = self.ctx.buffer(texcoords)
        self.ibo = self.ctx.buffer(indices)
        self.vao = self.ctx.vertex_array(self.program, [
            (self.vbo_vertices, '3f', 'in_position'),
            (self.vbo_texcoords, '2f', 'in_texcoord'),
        ], self.ibo)

    def _update_heightmap(self):
        """Generate new heightmap using PyFastFlow Perlin noise."""
        heightmap = pff.noise.perlin_noise(
            self.mesh_nx, self.mesh_ny,
            frequency=self.noise_params['frequency'],
            octaves=self.noise_params['octaves'],
            persistence=self.noise_params['persistence'],
            amplitude=self.noise_params['amplitude'],
            seed=self.noise_params['seed']
        )
        self.min_height = float(np.min(heightmap))
        self.max_height = float(np.max(heightmap))
        if self.max_height > self.min_height:
            heightmap_normalized = (heightmap - self.min_height) / (self.max_height - self.min_height)
        else:
            heightmap_normalized = np.zeros_like(heightmap)
        self.heightmap_cpu = heightmap_normalized.astype(np.float32)
        heightmap_bytes = (heightmap_normalized * 255).astype(np.uint8)
        if hasattr(self, 'heightmap_texture'):
            self.heightmap_texture.release()
        self.heightmap_texture = self.ctx.texture((self.mesh_nx, self.mesh_ny), 1, heightmap_bytes.tobytes())
        self.heightmap_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        # Prevent wrap-around at edges causing border artifacts
        self.heightmap_texture.repeat_x = False
        self.heightmap_texture.repeat_y = False

    def _world_to_grid(self, x: float, z: float) -> Optional[Tuple[float, float]]:
        nx, ny = self.mesh_nx, self.mesh_ny
        half = self.mesh_size * 0.5
        if x < -half or x > half or z < -half or z > half:
            return None
        u = (x / self.mesh_size + 0.5) * (nx - 1)
        v = (z / self.mesh_size + 0.5) * (ny - 1)
        return u, v

    def _sample_height(self, x: float, z: float) -> Optional[float]:
        if self.heightmap_cpu is None:
            return None
        nx, ny = self.mesh_nx, self.mesh_ny
        grid = self.heightmap_cpu
        uv = self._world_to_grid(x, z)
        if uv is None:
            return None
        u, v = uv
        i0 = int(math.floor(u))
        j0 = int(math.floor(v))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        du = u - i0
        dv = v - j0
        h00 = grid[j0, i0]
        h10 = grid[j0, i1]
        h01 = grid[j1, i0]
        h11 = grid[j1, i1]
        h0 = h00 * (1 - du) + h10 * du
        h1 = h01 * (1 - du) + h11 * du
        h = h0 * (1 - dv) + h1 * dv
        return float(h * self.height_scale)

    def intersect_ray(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> Optional[np.ndarray]:
        rd = ray_dir.astype(np.float32)
        n = np.linalg.norm(rd)
        if n == 0:
            return None
        rd /= n
        ro = ray_origin.astype(np.float32)
        t = 0.0
        t_max = self.mesh_size * 6.0
        dt = max(self.mesh_size / max(self.mesh_nx, self.mesh_ny), 0.002)
        prev_s = None
        prev_t = None
        while t <= t_max:
            p = ro + rd * t
            uv = self._world_to_grid(p[0], p[2])
            inside = uv is not None
            h = self._sample_height(p[0], p[2]) if inside else None
            s = p[1] - (h if h is not None else p[1] + 1.0)
            if inside and prev_s is not None and prev_s > 0.0 and s <= 0.0:
                a = prev_t
                b = t
                for _ in range(12):
                    m = 0.5 * (a + b)
                    pm = ro + rd * m
                    hm = self._sample_height(pm[0], pm[2])
                    sm = pm[1] - (hm if hm is not None else pm[1] + 1.0)
                    if sm > 0.0:
                        a = m
                    else:
                        b = m
                hit = ro + rd * 0.5 * (a + b)
                return hit
            if inside:
                prev_s = s
                prev_t = t
            t += dt * (1.0 + 0.5 * t / (self.mesh_size + 1e-6))
        return None

    def render(self, view_matrix: np.ndarray, projection_matrix: np.ndarray):
        """Render the terrain mesh."""
        # Bind heightmap texture
        self.heightmap_texture.use(0)
        # Set uniforms
        model_matrix = np.eye(4, dtype=np.float32)
        self.program['u_model'].write(model_matrix.T.astype('f4').tobytes())
        self.program['u_view'].write(view_matrix.T.astype('f4').tobytes())
        self.program['u_projection'].write(projection_matrix.T.astype('f4').tobytes())
        self.program['u_heightmap'] = 0
        self.program['u_height_scale'] = self.height_scale
        self.program['u_light_direction'].write(self.light_direction.tobytes())
        self.program['u_min_height'] = 0.0
        self.program['u_max_height'] = self.height_scale
        # Render the mesh
        self.vao.render()

    def update_noise_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.noise_params:
                self.noise_params[key] = value
        self._update_heightmap()

    # NOTE: The class previously had duplicate method definitions due to a bad merge.
    # The duplicates are removed and a single, consistent implementation is kept.

    def _generate_mesh(self):
        """Generate a 2D mesh with self.mesh_dim nodes on the longest dimension."""
        # Calculate mesh resolution (self.mesh_dim on longest dimension)
        aspect_ratio = self.width / self.height
        if self.width >= self.height:
            nx = self.mesh_dim
            ny = int(self.mesh_dim / aspect_ratio)
        else:
            nx = int(self.mesh_dim * aspect_ratio)
            ny = self.mesh_dim
            
        self.mesh_nx, self.mesh_ny = nx, ny
        
        # Generate mesh vertices
        vertices = []
        texcoords = []
        
        for j in range(ny):
            for i in range(nx):
                # Position in mesh space [-mesh_size/2, mesh_size/2]
                x = (i / (nx - 1) - 0.5) * self.mesh_size
                z = (j / (ny - 1) - 0.5) * self.mesh_size
                y = 0.0  # Will be displaced by heightmap in shader
                
                vertices.extend([x, y, z])
                texcoords.extend([i / (nx - 1), j / (ny - 1)])
        
        # Generate indices for triangles
        indices = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                # Current quad vertices
                tl = j * nx + i           # Top-left
                tr = j * nx + i + 1       # Top-right  
                bl = (j + 1) * nx + i     # Bottom-left
                br = (j + 1) * nx + i + 1 # Bottom-right
                
                # Two triangles per quad
                indices.extend([tl, bl, tr])  # First triangle
                indices.extend([tr, bl, br])  # Second triangle
        
        # Create vertex arrays and buffers
        vertices = np.array(vertices, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create vertex buffer objects
        self.vbo_vertices = self.ctx.buffer(vertices)
        self.vbo_texcoords = self.ctx.buffer(texcoords)
        self.ibo = self.ctx.buffer(indices)
        
        # Create vertex array object
        self.vao = self.ctx.vertex_array(self.program, [
            (self.vbo_vertices, '3f', 'in_position'),
            (self.vbo_texcoords, '2f', 'in_texcoord'),
        ], self.ibo)
        
    def _update_heightmap(self):
        """Generate new heightmap using PyFastFlow Perlin noise."""
        # Generate Perlin noise heightmap
        heightmap = pff.noise.perlin_noise(
            self.mesh_nx, self.mesh_ny,
            frequency=self.noise_params['frequency'],
            octaves=self.noise_params['octaves'],
            persistence=self.noise_params['persistence'],
            amplitude=self.noise_params['amplitude'],
            seed=self.noise_params['seed']
        )
        
        # Convert to [0, 1] range for texture
        self.min_height = float(np.min(heightmap))
        self.max_height = float(np.max(heightmap))
        
        if self.max_height > self.min_height:
            heightmap_normalized = (heightmap - self.min_height) / (self.max_height - self.min_height)
        else:
            heightmap_normalized = np.zeros_like(heightmap)
            
        # Keep CPU copy for ray intersection
        self.heightmap_cpu = heightmap_normalized.astype(np.float32)

        # Create/update heightmap texture
        heightmap_bytes = (heightmap_normalized * 255).astype(np.uint8)
        
        if hasattr(self, 'heightmap_texture'):
            self.heightmap_texture.release()
            
        # ModernGL expects (width, height); perlin returns (ny, nx), so provide raw bytes
        self.heightmap_texture = self.ctx.texture((self.mesh_nx, self.mesh_ny), 1, heightmap_bytes.tobytes())
        self.heightmap_texture.filter = moderngl.LINEAR, moderngl.LINEAR

    def _world_to_grid(self, x: float, z: float) -> Optional[Tuple[float, float]]:
        """Map world (x,z) to heightmap indices (u,v) in grid coordinates."""
        nx, ny = self.mesh_nx, self.mesh_ny
        half = self.mesh_size * 0.5
        if x < -half or x > half or z < -half or z > half:
            return None
        u = (x / self.mesh_size + 0.5) * (nx - 1)
        v = (z / self.mesh_size + 0.5) * (ny - 1)
        return u, v

    def _sample_height(self, x: float, z: float) -> Optional[float]:
        """Bilinear sample of CPU heightmap at world (x,z). Returns world y."""
        if self.heightmap_cpu is None:
            return None
        nx, ny = self.mesh_nx, self.mesh_ny
        grid = self.heightmap_cpu
        uv = self._world_to_grid(x, z)
        if uv is None:
            return None
        u, v = uv
        i0 = int(math.floor(u))
        j0 = int(math.floor(v))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        du = u - i0
        dv = v - j0
        h00 = grid[j0, i0]
        h10 = grid[j0, i1]
        h01 = grid[j1, i0]
        h11 = grid[j1, i1]
        h0 = h00 * (1 - du) + h10 * du
        h1 = h01 * (1 - du) + h11 * du
        h = h0 * (1 - dv) + h1 * dv
        return float(h * self.height_scale)

    def intersect_ray(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> Optional[np.ndarray]:
        """Ray-march from origin along dir to find intersection with terrain.
        Returns world-space point or None if no hit.
        """
        # Normalize direction
        rd = ray_dir.astype(np.float32)
        n = np.linalg.norm(rd)
        if n == 0:
            return None
        rd /= n
        ro = ray_origin.astype(np.float32)

        # March until we enter bounds; then look for crossing y - h(x,z) = 0
        t = 0.0
        t_max = self.mesh_size * 6.0
        dt = max(self.mesh_size / max(self.mesh_nx, self.mesh_ny), 0.002)
        prev_s = None
        prev_t = None

        while t <= t_max:
            p = ro + rd * t
            # If outside xz bounds, advance; once inside, proceed normally
            uv = self._world_to_grid(p[0], p[2])
            inside = uv is not None
            h = self._sample_height(p[0], p[2]) if inside else None
            s = p[1] - (h if h is not None else p[1] + 1.0)  # positive above surface

            if inside and prev_s is not None and prev_s > 0.0 and s <= 0.0:
                # Bracketed crossing between prev_t and t -> refine
                a = prev_t
                b = t
                for _ in range(12):
                    m = 0.5 * (a + b)
                    pm = ro + rd * m
                    hm = self._sample_height(pm[0], pm[2])
                    sm = pm[1] - (hm if hm is not None else pm[1] + 1.0)
                    if sm > 0.0:
                        a = m
                    else:
                        b = m
                hit = ro + rd * 0.5 * (a + b)
                return hit

            if inside:
                prev_s = s
                prev_t = t

            # Adaptive step: quicker when far away
            t += dt * (1.0 + 0.5 * t / (self.mesh_size + 1e-6))

        return None
        
    def update_noise_params(self, **kwargs):
        """Update noise parameters and regenerate heightmap."""
        for key, value in kwargs.items():
            if key in self.noise_params:
                self.noise_params[key] = value
        self._update_heightmap()


def create_projection_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create a perspective projection matrix."""
    f = 1.0 / math.tan(fov / 2.0)
    
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = (2.0 * far * near) / (near - far)
    result[3, 2] = -1.0
    
    return result


def main():
    """Main application loop."""
    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    # Set OpenGL context hints before window creation
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
    # Forward-compatible contexts can be problematic with some drivers/backends
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)
    
    # Create window
    width, height = 1200, 800
    window = glfw.create_window(width, height, "PyFastFlow Terrain Demo", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # V-sync
    
    # Initialize camera and renderer first (ModernGL wraps current context)
    camera = Camera(target=(0, 0, 0), distance=3.0, pitch=45.0, yaw=0.0)
    
    try:
        renderer = TerrainRenderer(width, height)
        print("TerrainRenderer initialized successfully")
    except Exception as e:
        print(f"Failed to initialize TerrainRenderer: {e}")
        glfw.terminate()
        raise

    # Initialize ImGui after ModernGL context is created
    imgui.create_context()
    io = imgui.get_io()
    # On some imgui versions this expects bytes, not None
    io.ini_file_name = b""
    # Build font atlas so new_frame assertions pass (we render later)
    io.fonts.get_tex_data_as_rgba32()
    # Force fully opaque widgets and visible scroll/slider grabs
    try:
        style = imgui.get_style()
        style.alpha = 1.0
        colors = style.colors
        # Frame/Window background high-contrast over terrain
        colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.07, 1.0)
        colors[imgui.COLOR_CHILD_BACKGROUND] = (0.06, 0.06, 0.07, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND] = (0.20, 0.22, 0.27, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.26, 0.59, 0.98, 1.0)
        colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = (0.26, 0.59, 0.98, 1.0)
        # Scrollbar + grab colors
        colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = (0.02, 0.02, 0.02, 1.0)
        colors[imgui.COLOR_SCROLLBAR_GRAB] = (0.31, 0.31, 0.31, 1.0)
        colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = (0.41, 0.41, 0.41, 1.0)
        colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = (0.51, 0.51, 0.51, 1.0)
        # Slider grab colors
        colors[imgui.COLOR_SLIDER_GRAB] = (0.24, 0.52, 0.88, 1.0)
        colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = (0.26, 0.59, 0.98, 1.0)
        # Text visible
        colors[imgui.COLOR_TEXT] = (1.00, 1.00, 1.00, 1.0)
    except Exception:
        pass
    # Minimal IO pump (no PyOpenGL backends to avoid segfaults)
    class ImGuiIOPump:
        def __init__(self):
            self.scroll_x = 0.0
            self.scroll_y = 0.0
            self._last_time = None
        def add_scroll(self, xoff, yoff):
            self.scroll_x += xoff
            self.scroll_y += yoff
        def process(self, win):
            io = imgui.get_io()
            # Delta time
            now = glfw.get_time()
            if self._last_time is None:
                io.delta_time = 1.0 / 60.0
            else:
                io.delta_time = max(1e-6, now - self._last_time)
            self._last_time = now
            # Use framebuffer size to avoid HiDPI issues
            fb_w, fb_h = glfw.get_framebuffer_size(win)
            io.display_size = (float(fb_w), float(fb_h))
            io.display_fb_scale = (1.0, 1.0)
            mx, my = glfw.get_cursor_pos(win)
            io.mouse_pos = (float(mx), float(my))
            io.mouse_down[0] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            io.mouse_down[1] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            io.mouse_down[2] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
            io.mouse_wheel = self.scroll_y
            io.mouse_wheel_horizontal = self.scroll_x
            self.scroll_x = 0.0
            self.scroll_y = 0.0
    io_pump = ImGuiIOPump()
    # Prefer the pyimgui ModernGL renderer (no PyOpenGL fixed pipeline needed)
    imgui_renderer = None
    imgui_renderer_kind = None
    renderer_init_errors = []
    try:
        # Provide a custom ModernGL ImGui renderer to avoid backend fragility
        raise ImportError("force_custom_imgui_renderer")
    except Exception as e:
        renderer_init_errors.append(("custom_renderer_setup", e))
        # Minimal self-contained ModernGL renderer for ImGui draw data
        class SimpleImGuiRenderer:
            def __init__(self, ctx: moderngl.Context):
                self.ctx = ctx
                self.prog = ctx.program(
                    vertex_shader='''
                        #version 330
                        uniform mat4 ProjMtx;
                        in vec2 Position;
                        in vec2 UV;
                        in vec4 Color;
                        out vec2 Frag_UV;
                        out vec4 Frag_Color;
                        void main() {
                            Frag_UV = UV;
                            Frag_Color = Color;
                            gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
                        }
                    ''',
                    fragment_shader='''
                        #version 330
                        uniform sampler2D Texture;
                        in vec2 Frag_UV;
                        in vec4 Frag_Color;
                        out vec4 Out_Color;
                        void main() {
                            Out_Color = Frag_Color * texture(Texture, Frag_UV);
                        }
                    ''')
                self.vbo = ctx.buffer(reserve=1)
                self.ibo = ctx.buffer(reserve=1)
                self.vao = ctx.vertex_array(
                    self.prog,
                    [
                        (self.vbo, '2f 2f 4f', 'Position', 'UV', 'Color')
                    ],
                    index_buffer=self.ibo,
                )
                # Font texture
                width, height, pixels = io.fonts.get_tex_data_as_rgba32()
                self.font_tex = ctx.texture((width, height), 4, pixels)
                self.font_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                self.font_tex.use(0)
                # Do not rely on ImGui texture_id; we always bind our font texture
                self.prog['Texture'].value = 0

            def render(self, draw_data):
                if draw_data is None:
                    return
                io = imgui.get_io()
                disp_w, disp_h = io.display_size
                fb_scale_x, fb_scale_y = io.display_fb_scale
                fb_w = int(disp_w * fb_scale_x)
                fb_h = int(disp_h * fb_scale_y)
                if fb_w == 0 or fb_h == 0:
                    return
                # Setup state
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.ctx.disable(moderngl.CULL_FACE)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.enable(moderngl.SCISSOR_TEST)
                # Projection (origin at top-left)
                L, R = 0.0, disp_w
                T, B = 0.0, disp_h
                proj = np.array([
                    [ 2.0/(R-L),      0.0,           0.0,  -(R+L)/(R-L)],
                    [      0.0,  2.0/(T-B),         0.0,  -(T+B)/(T-B)],
                    [      0.0,      0.0,          -1.0,            0.0],
                    [      0.0,      0.0,           0.0,            1.0],
                ], dtype=np.float32)
                self.prog['ProjMtx'].write(proj.T.tobytes())

                # Build buffers
                vertices = []
                indices = []
                idx_base = 0
                # Access command lists (support both API styles)
                try:
                    cmd_lists = draw_data.commands_lists
                    if callable(cmd_lists):
                        cmd_lists = cmd_lists()
                except AttributeError:
                    try:
                        cmd_lists = draw_data.cmd_lists()
                    except Exception:
                        cmd_lists = []
                # Build combined buffers
                for cl in cmd_lists:
                    # Vertices
                    for v in cl.vtx_buffer:
                        x = float(getattr(v.pos, 'x', getattr(v.pos, '__getitem__', lambda i: v.pos)[0]))
                        y = float(getattr(v.pos, 'y', getattr(v.pos, '__getitem__', lambda i: v.pos)[1]))
                        u = float(getattr(v.uv, 'x', getattr(v.uv, '__getitem__', lambda i: v.uv)[0]))
                        vv = float(getattr(v.uv, 'y', getattr(v.uv, '__getitem__', lambda i: v.uv)[1]))
                        c = int(v.col)
                        r = (c & 0xFF) / 255.0
                        g = ((c >> 8) & 0xFF) / 255.0
                        b = ((c >> 16) & 0xFF) / 255.0
                        a = ((c >> 24) & 0xFF) / 255.0
                        vertices.extend([x, y, u, vv, r, g, b, a])
                    # Indices
                    for idx in cl.idx_buffer:
                        indices.append(int(idx) + idx_base)
                    idx_base += len(cl.vtx_buffer)

                vtx_arr = np.array(vertices, dtype=np.float32)
                # Use 32-bit indices to match ModernGL's default index_element_size=4
                # Avoid mismatches that can make UI invisible
                idx_arr = np.array(indices, dtype=np.uint32)
                self.vbo.orphan(vtx_arr.nbytes)
                self.vbo.write(vtx_arr.tobytes())
                self.ibo.orphan(idx_arr.nbytes)
                self.ibo.write(idx_arr.tobytes())

                # Draw by commands with scissor
                idx_global = 0
                for cl in cmd_lists:
                    cmd_list_commands = getattr(cl, 'commands', None)
                    if callable(cmd_list_commands):
                        cmd_iter = cmd_list_commands()
                    elif cmd_list_commands is not None:
                        cmd_iter = cmd_list_commands
                    else:
                        try:
                            cmd_iter = cl.cmd_buffer
                        except Exception:
                            cmd_iter = []
                    for cmd in cmd_iter:
                        x1, y1, x2, y2 = cmd.clip_rect
                        # Apply framebuffer scale, convert to scissor with origin at bottom-left
                        x = int(x1 * fb_scale_x)
                        y = int((disp_h - y2) * fb_scale_y)
                        w = int((x2 - x1) * fb_scale_x)
                        h = int((y2 - y1) * fb_scale_y)
                        if w <= 0 or h <= 0:
                            idx_global += cmd.elem_count
                            continue
                        self.ctx.scissor = (x, y, w, h)
                        # Bind font texture
                        self.font_tex.use(0)
                        self.vao.render(moderngl.TRIANGLES, vertices=cmd.elem_count, first=idx_global)
                        idx_global += cmd.elem_count

                # Restore state
                self.ctx.disable(moderngl.SCISSOR_TEST)
                self.ctx.disable(moderngl.BLEND)

        imgui_renderer = SimpleImGuiRenderer(renderer.ctx)
        imgui_renderer_kind = 'custom'
        print("Custom ImGui ModernGL renderer initialized")

    ui_state = {
        'live_noise_update': False,
    }
    
    # Fullscreen toggle helpers
    state = {
        'fullscreen': False,
        'win_x': 100,
        'win_y': 100,
        'win_w': width,
        'win_h': height,
    }

    def set_fullscreen(enable: bool):
        if enable == state['fullscreen']:
            return
        state['fullscreen'] = enable
        if enable:
            # Save windowed position/size
            x, y = glfw.get_window_pos(window)
            w, h = glfw.get_window_size(window)
            state.update({'win_x': x, 'win_y': y, 'win_w': w, 'win_h': h})
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            if monitor and mode:
                glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            else:
                # Borderless fullscreen fallback
                glfw.set_window_attrib(window, glfw.DECORATED, False)
                glfw.set_window_pos(window, 0, 0)
                glfw.set_window_size(window, state['win_w'], state['win_h'])
        else:
            glfw.set_window_monitor(window, None, state['win_x'], state['win_y'], state['win_w'], state['win_h'], 0)
            glfw.set_window_attrib(window, glfw.DECORATED, True)

    # Set up GLFW callbacks
    def mouse_button_callback(window, button, action, mods):
        # Respect ImGui capture only if we actually render UI
        io = imgui.get_io()
        if (imgui_renderer is not None) and io.want_capture_mouse:
            return
        camera.handle_mouse(window, button, action, mods)
    
    def cursor_pos_callback(window, xpos, ypos):
        io = imgui.get_io()
        if (imgui_renderer is not None) and io.want_capture_mouse:
            return
        camera.handle_cursor(window, xpos, ypos)
    
    def scroll_callback(window, xoffset, yoffset):
        io_pump.add_scroll(xoffset, yoffset)
        io = imgui.get_io()
        if (imgui_renderer is not None) and io.want_capture_mouse:
            return
        # Compute mouse ray in world space
        mx, my = glfw.get_cursor_pos(window)
        vw, vh = width, height
        view_m = camera.get_view_matrix()

        # Build inverse of projection*view
        vp = projection_matrix @ view_m
        try:
            inv_vp = np.linalg.inv(vp)
        except np.linalg.LinAlgError:
            # Fallback to simple zoom if matrix not invertible
            camera.handle_scroll(window, xoffset, yoffset)
            return

        x_ndc = (2.0 * mx / vw) - 1.0
        y_ndc = 1.0 - (2.0 * my / vh)
        near_ndc = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float32)
        far_ndc = np.array([x_ndc, y_ndc, 1.0, 1.0], dtype=np.float32)

        near_world = inv_vp @ near_ndc
        far_world = inv_vp @ far_ndc
        if near_world[3] != 0:
            near_world = near_world / near_world[3]
        if far_world[3] != 0:
            far_world = far_world / far_world[3]

        origin = camera.get_eye()
        direction = (far_world[:3] - near_world[:3]).astype(np.float32)
        n = np.linalg.norm(direction)
        if n == 0:
            camera.handle_scroll(window, xoffset, yoffset)
            return
        direction /= n

        hit = renderer.intersect_ray(origin, direction)

        # Compute zoom scale (dolly) and adjust target towards hit point
        zoom_speed = 0.18
        s = math.exp(-yoffset * zoom_speed)
        s = max(0.1, min(10.0, s))

        if hit is not None:
            camera.target = camera.target + (1.0 - s) * (hit - camera.target)
        # Update distance multiplicatively and clamp
        camera.distance = max(0.5, camera.distance * s)
    
    def key_callback(window, key, scancode, action, mods):
        # Fullscreen toggle
        if key == glfw.KEY_F11 and action == glfw.PRESS:
            set_fullscreen(not state['fullscreen'])
            return
        if key == glfw.KEY_ENTER and (mods & glfw.MOD_ALT) and action == glfw.PRESS:
            set_fullscreen(not state['fullscreen'])
            return
        # Let ESC always close; otherwise, if ImGui wants keyboard, skip
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            return
        io = imgui.get_io()
        if (imgui_renderer is not None) and io.want_capture_keyboard:
            return
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    # If GlfwRenderer installed its own callbacks, chain ours after it so camera works
    if imgui_renderer_kind == 'glfw':
        try:
            prev_mouse_cb = glfw.get_mouse_button_callback(window)
            prev_cursor_cb = glfw.get_cursor_pos_callback(window)
            prev_scroll_cb = glfw.get_scroll_callback(window)
            prev_key_cb = glfw.get_key_callback(window)

            def chained_mouse_btn(win, button, action, mods):
                if prev_mouse_cb:
                    prev_mouse_cb(win, button, action, mods)
                mouse_button_callback(win, button, action, mods)

            def chained_cursor(win, x, y):
                if prev_cursor_cb:
                    prev_cursor_cb(win, x, y)
                cursor_pos_callback(win, x, y)

            def chained_scroll(win, xoff, yoff):
                if prev_scroll_cb:
                    prev_scroll_cb(win, xoff, yoff)
                scroll_callback(win, xoff, yoff)

            def chained_key(win, key, scancode, action, mods):
                if prev_key_cb:
                    prev_key_cb(win, key, scancode, action, mods)
                key_callback(win, key, scancode, action, mods)

            glfw.set_mouse_button_callback(window, chained_mouse_btn)
            glfw.set_cursor_pos_callback(window, chained_cursor)
            glfw.set_scroll_callback(window, chained_scroll)
            glfw.set_key_callback(window, chained_key)
        except Exception:
            pass
    
    # Initial projection (will be updated per-frame)
    projection_matrix = create_projection_matrix(math.radians(60.0), max(1.0, width / height), 0.1, 100.0)
    
    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if imgui_renderer_kind == 'glfw' and hasattr(imgui_renderer, 'process_inputs'):
            try:
                imgui_renderer.process_inputs()
            except Exception:
                pass
        elif imgui_renderer_kind != 'glfw':
            io_pump.process(window)
        imgui.new_frame()
        
        # Update viewport and projection to current framebuffer size
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        fb_w = max(1, fb_w)
        fb_h = max(1, fb_h)
        renderer.ctx.viewport = (0, 0, fb_w, fb_h)
        projection_matrix = create_projection_matrix(math.radians(60.0), fb_w / fb_h, 0.1, 100.0)

        # Clear screen (color and depth)
        try:
            renderer.ctx.clear(0.2, 0.3, 0.4, 1.0, depth=1.0)
        except TypeError:
            renderer.ctx.clear(0.2, 0.3, 0.4)  # Fallback for older moderngl
        
        # Render terrain
        view_matrix = camera.get_view_matrix()
        renderer.render(view_matrix, projection_matrix)

        # ImGui UI
        # Demo window to ensure something visible (can be removed later)
        imgui.show_demo_window()
        if glfw.get_time() < 1.0:
            imgui.set_next_window_position(10.0, 10.0)
            imgui.set_next_window_size(360.0, 300.0)
        imgui.begin("Terrain Controls")
        changed = False
        f = renderer.noise_params['frequency']
        changed_f, f = imgui.slider_float("frequency", f, 1.0, 64.0, '%.1f')
        o = renderer.noise_params['octaves']
        changed_o, o = imgui.slider_int("octaves", int(o), 1, 10)
        p = renderer.noise_params['persistence']
        changed_p, p = imgui.slider_float("persistence", p, 0.1, 0.95, '%.2f')
        a = renderer.noise_params['amplitude']
        changed_a, a = imgui.slider_float("amplitude", a, 0.1, 5.0, '%.2f')
        hs = renderer.height_scale
        changed_hs, hs = imgui.slider_float("Z exaggeration", hs, 0.01, 5.0, '%.2f')
        if changed_f or changed_o or changed_p or changed_a:
            changed = True
        if changed_hs:
            renderer.height_scale = float(hs)
        # Live or manual apply for noise changes
        imgui.separator()
        live = ui_state['live_noise_update']
        changed_live, live = imgui.checkbox("Live noise update", live)
        if changed_live:
            ui_state['live_noise_update'] = live
        imgui.same_line()
        if imgui.button("Apply noise changes") and changed:
            renderer.update_noise_params(frequency=float(f), octaves=int(o), persistence=float(p), amplitude=float(a))
            changed = False
        if ui_state['live_noise_update'] and changed:
            renderer.update_noise_params(frequency=float(f), octaves=int(o), persistence=float(p), amplitude=float(a))
            changed = False
        lx, ly, lz = tuple(renderer.light_direction.tolist())
        changed_lx, lx = imgui.slider_float("light_x", lx, -1.0, 1.0, '%.2f')
        changed_ly, ly = imgui.slider_float("light_y", ly, 0.1, 2.0, '%.2f')
        changed_lz, lz = imgui.slider_float("light_z", lz, -1.0, 1.0, '%.2f')
        if changed_lx or changed_ly or changed_lz:
            v = np.array([lx, ly, lz], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                renderer.light_direction = v / n
        if imgui.button("New Seed"):
            renderer.noise_params['seed'] = int((renderer.noise_params['seed'] + 1) % 1000000)
            renderer.update_noise_params()
        imgui.separator()
        imgui.text(f"Camera dist: {camera.distance:.2f}")
        imgui.text(f"Yaw/Pitch: {camera.yaw:.1f}/{camera.pitch:.1f}")
        imgui.text(f"FPS: {1.0/max(glfw.get_time(),1e-6):.0f}")
        # Fullscreen toggle
        fs = state['fullscreen']
        changed_fs, fs = imgui.checkbox("Fullscreen (F11)", fs)
        if changed_fs:
            set_fullscreen(fs)
        imgui.end()

        # Always-visible overlay with Z exaggeration slider (top-left)
        imgui.set_next_window_bg_alpha(1.0)
        imgui.set_next_window_position(10.0, 10.0)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS)
        imgui.begin("ZExOverlay", flags=flags)
        imgui.text("UI DEBUG: If you read this, UI draws.")
        hs_overlay = renderer.height_scale
        changed_hs2, hs_overlay = imgui.slider_float("Z exaggeration", hs_overlay, 0.01, 5.0, '%.2f')
        if changed_hs2:
            renderer.height_scale = float(hs_overlay)
        imgui.end()

        imgui.render()
        if imgui_renderer is not None:
            # Ensure viewport and blend/depth state for overlay
            fb_w, fb_h = glfw.get_framebuffer_size(window)
            renderer.ctx.viewport = (0, 0, max(1, fb_w), max(1, fb_h))
            # Render overlay
            dd = imgui.get_draw_data()
            try:
                imgui_renderer.render(dd)
                if getattr(dd, 'total_vtx_count', 0) == 0:
                    if 'warn_no_vtx' not in ui_state:
                        ui_state['warn_no_vtx'] = True
                        print("WARN: ImGui draw_data has zero vertices; UI may be invisible.")
            except Exception as e:
                if 'ui_render_error' not in ui_state:
                    ui_state['ui_render_error'] = True
                    print("ImGui render error:", e)
        
        # Print controls info once
        if glfw.get_time() < 0.1:  # Only print at startup
            print("Controls:")
            print("- Left Mouse: Rotate view")
            print("- Right Mouse: Pan center")
            print("- Mouse Wheel: Zoom")
            print("- ESC: Exit")
        
        glfw.swap_buffers(window)
    
    # Cleanup
    try:
        if imgui_renderer is not None and hasattr(imgui_renderer, 'shutdown'):
            imgui_renderer.shutdown()
    except Exception:
        pass
    glfw.terminate()


if __name__ == "__main__":
    main()
