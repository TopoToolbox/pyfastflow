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
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer
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

# ImGui compatibility flags (older pyimgui may not expose these)
IMGUI_COND_ALWAYS = getattr(imgui, 'COND_ALWAYS', 0)
IMGUI_COND_APPEARING = getattr(imgui, 'COND_APPEARING', getattr(imgui, 'COND_FIRST_USE_EVER', 0))


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
    
    def __init__(self, width: int, height: int, ctx: Optional[moderngl.Context] = None):
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
        self._init_gl(ctx)
        self._create_shaders()
        self._generate_mesh()
        self._update_heightmap()

    def _init_gl(self, ctx: Optional[moderngl.Context] = None):
        """Initialize OpenGL context via ModernGL."""
        self.ctx = ctx or moderngl.create_context()
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
        uniform float u_hide_underside;

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
            // Optionally hide back-facing primitives regardless of normal correctness
            if (u_hide_underside > 0.5 && !gl_FrontFacing) {
                discard;
            }
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

    def render(self, view_matrix: np.ndarray, projection_matrix: np.ndarray, eye: np.ndarray, hide_underside: bool = True):
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
        try:
            self.program['u_hide_underside'] = 1.0 if hide_underside else 0.0
        except Exception:
            pass
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

class TerrainApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "PyFastFlow Terrain (moderngl-window + ImGui)"
    window_size = (1200, 800)
    aspect_ratio = None
    resource_dir = '.'
    # Force GLFW backend for robust mouse input
    window = 'glfw'
    vsync = True
    cursor = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        w, h = self.wnd.buffer_size
        self.camera = Camera(target=(0, 0, 0), distance=3.0, pitch=45.0, yaw=0.0)
        self.renderer = TerrainRenderer(w, h, ctx=self.ctx)
        self.live_noise_update = False
        self.hide_underside = False
        self._received_motion = False
        self._last_event_pos = (0.0, 0.0)
        self._evt_counts = {
            'pos': 0,
            'drag': 0,
            'press': 0,
            'release': 0,
            'scroll': 0,
        }
        self._last_evt_dxdy = (0.0, 0.0)
        # Allow toggling relative mouse (cursor lock) from UI
        self.relative_mouse = False
        try:
            self.wnd.mouse_exclusivity = False
        except Exception:
            pass
        # Initialize last mouse position with a finite value
        try:
            mp = tuple(self.wnd.mouse_position)
            if len(mp) == 2 and all(np.isfinite(v) for v in mp):
                self._last_mouse_pos = mp
            else:
                self._last_mouse_pos = (0.0, 0.0)
        except Exception:
            self._last_mouse_pos = (0.0, 0.0)

        # Explicitly hook into moderngl-window's event function slots to ensure we get events
        try:
            self.wnd.mouse_position_event_func = self._evt_mouse_position
            self.wnd.mouse_drag_event_func = self._evt_mouse_drag
            self.wnd.mouse_press_event_func = self._evt_mouse_press
            self.wnd.mouse_release_event_func = self._evt_mouse_release
            self.wnd.mouse_scroll_event_func = self._evt_mouse_scroll
        except Exception:
            pass

    # Direct event hooks (bypass WindowConfig callbacks if they are not invoked)
    def _evt_mouse_press(self, x: int, y: int, button: int):
        try:
            self.imgui.mouse_press_event(x, y, button)
        except Exception:
            pass
        self._evt_counts['press'] += 1
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        if not self.imgui.io.want_capture_mouse:
            # nothing else to do on press
            pass

    def _evt_mouse_release(self, x: int, y: int, button: int):
        try:
            self.imgui.mouse_release_event(x, y, button)
        except Exception:
            pass
        self._evt_counts['release'] += 1
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))

    def _evt_mouse_drag(self, x: int, y: int, dx: int, dy: int):
        try:
            self.imgui.mouse_drag_event(x, y, dx, dy)
        except Exception:
            pass
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        if all(np.isfinite(v) for v in (dx, dy)):
            self._last_evt_dxdy = (float(dx), float(dy))
        self._evt_counts['drag'] += 1
        if not self.imgui.io.want_capture_mouse:
            self._apply_camera_drag(dx, dy)

    def _evt_mouse_position(self, x: int, y: int, dx: int, dy: int):
        try:
            self.imgui.mouse_position_event(x, y, dx, dy)
        except Exception:
            pass
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        if all(np.isfinite(v) for v in (dx, dy)):
            self._last_evt_dxdy = (float(dx), float(dy))
        self._evt_counts['pos'] += 1
        if not self.imgui.io.want_capture_mouse:
            self._apply_camera_drag(dx, dy)

    def _evt_mouse_scroll(self, x_offset: float, y_offset: float):
        try:
            self.imgui.mouse_scroll_event(x_offset, y_offset)
        except Exception:
            pass
        self._evt_counts['scroll'] += 1
        if not self.imgui.io.want_capture_mouse:
            self.camera.handle_scroll(None, x_offset, y_offset)

    def _get_glfw_window(self):
        # Try common attribute names for the underlying GLFW window
        for name in ('_window', '_native_window', 'window'):
            try:
                win = getattr(self.wnd, name, None)
                if win is not None:
                    return win
            except Exception:
                pass
        try:
            # Some versions expose through context
            return getattr(getattr(self.wnd, 'ctx', None), 'window', None)
        except Exception:
            return None

    def resize(self, width: int, height: int):
        # Window size in screen coordinates
        self.ctx.viewport = (0, 0, max(1, width), max(1, height))
        # Inform ImGui renderer so it updates display size and fb scale
        try:
            self.imgui.resize(width, height)
            self._update_imgui_display(width, height)
        except Exception:
            pass

    # Some systems fire a framebuffer resize separate from window resize
    def buffer_size_event(self, width: int, height: int):
        try:
            self.ctx.viewport = (0, 0, max(1, width), max(1, height))
            # Update imgui framebuffer scale
            try:
                win_w, win_h = self.wnd.size
            except Exception:
                win_w, win_h = width, height
            self._update_imgui_display(win_w, win_h, width, height)
        except Exception:
            pass

    def _update_imgui_display(self, win_w: int, win_h: int, fb_w: int | None = None, fb_h: int | None = None):
        try:
            io = imgui.get_io()
            io.display_size = (float(max(1, win_w)), float(max(1, win_h)))
            if fb_w is None or fb_h is None:
                try:
                    fb_w, fb_h = self.wnd.buffer_size
                except Exception:
                    fb_w, fb_h = win_w, win_h
            sx = float(max(1, fb_w)) / float(max(1, win_w))
            sy = float(max(1, fb_h)) / float(max(1, win_h))
            io.display_fb_scale = (sx, sy)
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
        # Optionally enable relative mouse for non-UI interaction
        try:
            if self.relative_mouse and button in (self.wnd.mouse.left, self.wnd.mouse.right, self.wnd.mouse.middle):
                self.wnd.mouse_exclusivity = True
        except Exception:
            pass

    def mouse_release_event(self, x: int, y: int, button: int):
        try:
            self.imgui.mouse_release_event(x, y, button)
        except Exception:
            pass
        self._evt_counts['release'] += 1
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        try:
            if self.relative_mouse:
                self.wnd.mouse_exclusivity = False
        except Exception:
            pass

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        try:
            self.imgui.mouse_drag_event(x, y, dx, dy)
        except Exception:
            pass
        # Track event-based motion and last event position
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        self._received_motion = True
        if all(np.isfinite(v) for v in (dx, dy)):
            self._last_evt_dxdy = (float(dx), float(dy))
        self._evt_counts['drag'] += 1
        if not self.imgui.io.want_capture_mouse:
            self._apply_camera_drag(dx, dy)

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        # Some backends report continuous motion here instead of drag_event
        try:
            self.imgui.mouse_position_event(x, y, dx, dy)
        except Exception:
            pass
        # Track event-based motion and last event position
        if all(np.isfinite(v) for v in (x, y)):
            self._last_event_pos = (float(x), float(y))
        self._received_motion = True
        if all(np.isfinite(v) for v in (dx, dy)):
            self._last_evt_dxdy = (float(dx), float(dy))
        self._evt_counts['pos'] += 1
        if not self.imgui.io.want_capture_mouse:
            self._apply_camera_drag(dx, dy)

    def _apply_camera_drag(self, dx: float, dy: float):
        ms = getattr(self.wnd, 'mouse_states', None)
        if not ms:
            return
        # Fallback: if dx,dy look unusable, try to compute from last event / raw cursor
        if not (np.isfinite(dx) and np.isfinite(dy)) or (dx == 0 and dy == 0 and (getattr(ms, 'left', False) or getattr(ms, 'right', False) or getattr(ms, 'middle', False))):
            # Try raw glfw cursor
            try:
                raw_win = getattr(self.wnd, '_window', None)
                cx, cy = (None, None)
                if raw_win is not None:
                    cx, cy = glfw.get_cursor_pos(raw_win)
                if cx is None or not (np.isfinite(cx) and np.isfinite(cy)):
                    cx, cy = self._last_event_pos
                if all(np.isfinite(v) for v in (cx, cy)) and all(np.isfinite(v) for v in self._last_event_pos):
                    dx = float(cx - self._last_event_pos[0])
                    dy = float(cy - self._last_event_pos[1])
                    self._last_event_pos = (float(cx), float(cy))
            except Exception:
                pass
        sensitivity = 0.5
        if getattr(ms, 'left', False):
            self.camera.yaw += dx * sensitivity
            self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch - dy * sensitivity))
        elif getattr(ms, 'right', False) or getattr(ms, 'middle', False):
            pitch_rad = math.radians(self.camera.pitch)
            yaw_rad = math.radians(self.camera.yaw)
            right = np.array([-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)], dtype=np.float32)
            forward = np.array([
                math.cos(pitch_rad) * math.cos(yaw_rad),
                math.sin(pitch_rad),
                math.cos(pitch_rad) * math.sin(yaw_rad),
            ], dtype=np.float32)
            right /= max(1e-6, np.linalg.norm(right))
            forward /= max(1e-6, np.linalg.norm(forward))
            up = np.cross(right, forward)
            up /= max(1e-6, np.linalg.norm(up))
            pan_speed = 0.01 * self.camera.distance
            self.camera.target += right * dx * pan_speed
            self.camera.target += up * (-dy) * pan_speed

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        try:
            self.imgui.mouse_scroll_event(x_offset, y_offset)
        except Exception:
            pass
        self._evt_counts['scroll'] += 1
        # Perform zoom regardless of ImGui capture to match the old behavior
        # Use ImGui IO for consistent window coordinates and scaling
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
            # Fallback to window mouse position
            try:
                mx2, my2 = self.wnd.mouse_position
                if np.isfinite(mx2) and np.isfinite(my2):
                    mx, my = float(mx2), float(my2)
            except Exception:
                pass
        if not (np.isfinite(mx) and np.isfinite(my)):
            # Fallback to center
            mx, my = win_w * 0.5, win_h * 0.5

        # Build projection/view if missing
        proj = getattr(self, '_last_proj', None)
        view = getattr(self, '_last_view', None)
        if proj is None or view is None:
            aspect = max(1.0, win_w / max(1.0, win_h))
            proj = create_projection_matrix(math.radians(60.0), aspect, 0.1, 100.0)
            view = self.camera.get_view_matrix()

        vp = proj @ view
        try:
            inv_vp = np.linalg.inv(vp)
        except Exception:
            inv_vp = None

        hit = None
        if inv_vp is not None:
            # Window coords -> NDC using window size
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
            origin = self.camera.get_eye()
            direction = (far_world[:3] - near_world[:3]).astype(np.float32)
            n = np.linalg.norm(direction)
            if n > 0:
                direction /= n
                hit = self.renderer.intersect_ray(origin, direction)
                # Intersect with plane y=0 if no terrain hit
                if hit is None and abs(direction[1]) > 1e-6:
                    t = -origin[1] / direction[1]
                    if t > 0:
                        hit = origin + direction * t

        # Dolly towards hit point
        zoom_speed = 0.18
        s = math.exp(-y_offset * zoom_speed)
        s = max(0.1, min(10.0, s))
        if hit is not None:
            self.camera.target = self.camera.target + (1.0 - s) * (hit - self.camera.target)
        # Update camera distance
        self.camera.distance = max(0.5, self.camera.distance * s)

    def key_event(self, key, action, modifiers):
        try:
            self.imgui.key_event(key, action, modifiers)
        except Exception:
            pass
        if not self.imgui.io.want_capture_keyboard:
            # Toggle fullscreen on F11
            if key == getattr(self.wnd.keys, 'F11', None) and action == self.wnd.keys.ACTION_PRESS:
                try:
                    self.wnd.fullscreen = not self.wnd.fullscreen
                except Exception:
                    pass
            if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_PRESS:
                self.wnd.close()

    def unicode_char_entered(self, char: str):
        try:
            self.imgui.unicode_char_entered(char)
        except Exception:
            pass

    # moderngl-window 3.x expects on_render
    def on_render(self, time: float, frame_time: float):
        fb_w, fb_h = self.wnd.buffer_size
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        try:
            self.ctx.clear(0.2, 0.3, 0.4, 1.0, depth=1.0)
        except TypeError:
            self.ctx.clear(0.2, 0.3, 0.4)
        proj = create_projection_matrix(math.radians(60.0), max(1.0, fb_w / max(1.0, fb_h)), 0.1, 100.0)
        view = self.camera.get_view_matrix()
        self._last_proj = proj
        self._last_view = view
        eye = self.camera.get_eye()
        self.renderer.render(view, proj, eye, self.hide_underside)

        # Update ImGui IO and start a new frame
        # If backend hasn't reported any mouse motion yet, try GLFW cursor pos as a fallback
        try:
            if (self._evt_counts['pos'] + self._evt_counts['drag']) == 0 and 'glfw' in type(self.wnd).__module__:
                raw_win = self._get_glfw_window()
                if raw_win is not None:
                    cx, cy = glfw.get_cursor_pos(raw_win)
                    if all(np.isfinite(v) for v in (cx, cy)):
                        self._last_event_pos = (float(cx), float(cy))
                        self._last_mouse_pos = (float(cx), float(cy))
        except Exception:
            pass
        try:
            # Ensure ImGui knows the current window and framebuffer size
            win_w, win_h = self.wnd.size
            self._update_imgui_display(win_w, win_h, fb_w, fb_h)
            self.imgui.process_inputs()
        except Exception:
            pass
        # Override bad imgui io mouse state using event-tracked values
        io_fix = imgui.get_io()
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
        imgui.set_next_window_size(360.0, 320.0)
        imgui.begin("Terrain Controls")
        changed = False
        f = self.renderer.noise_params['frequency']
        ch_f, f = imgui.slider_float("frequency", f, 1.0, 64.0, '%.1f')
        o = int(self.renderer.noise_params['octaves'])
        ch_o, o = imgui.slider_int("octaves", o, 1, 10)
        p = self.renderer.noise_params['persistence']
        ch_p, p = imgui.slider_float("persistence", p, 0.1, 0.95, '%.2f')
        a = self.renderer.noise_params['amplitude']
        ch_a, a = imgui.slider_float("amplitude", a, 0.1, 5.0, '%.2f')
        hs = self.renderer.height_scale
        ch_hs, hs = imgui.slider_float("Z exaggeration", hs, 0.01, 5.0, '%.2f')
        if ch_f or ch_o or ch_p or ch_a:
            changed = True
        if ch_hs:
            self.renderer.height_scale = float(hs)
        imgui.separator()
        if imgui.checkbox("Live noise update", self.live_noise_update)[0]:
            self.live_noise_update = not self.live_noise_update
        imgui.same_line()
        if imgui.button("Apply noise changes") and changed:
            self.renderer.update_noise_params(
                frequency=float(f), octaves=int(o), persistence=float(p), amplitude=float(a)
            )
            changed = False
        if self.live_noise_update and changed:
            self.renderer.update_noise_params(
                frequency=float(f), octaves=int(o), persistence=float(p), amplitude=float(a)
            )
            changed = False
        lx, ly, lz = tuple(self.renderer.light_direction.tolist())
        ch_lx, lx = imgui.slider_float("light_x", lx, -1.0, 1.0, '%.2f')
        ch_ly, ly = imgui.slider_float("light_y", ly, 0.1, 2.0, '%.2f')
        ch_lz, lz = imgui.slider_float("light_z", lz, -1.0, 1.0, '%.2f')
        if ch_lx or ch_ly or ch_lz:
            v = np.array([lx, ly, lz], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                self.renderer.light_direction = v / n
        imgui.text(f"FPS: {1.0/max(frame_time,1e-6):.0f}")
        ch_hide, new_hide = imgui.checkbox("Hide underside (front-facing only)", self.hide_underside)
        if ch_hide:
            self.hide_underside = bool(new_hide)
        imgui.end()

        imgui.set_next_window_bg_alpha(1.0)
        imgui.set_next_window_position(10.0, 10.0)
        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS)
        imgui.begin("ZExOverlay", flags=flags)
        io_dbg = imgui.get_io()
        imgui.text("UI DEBUG")
        # Move key debug lines earlier so they are visible even in small windows
        imgui.text(f"want_capture_mouse={bool(io_dbg.want_capture_mouse)}")
        imgui.text(f"evt_counts pos/drag/press/release/scroll = {self._evt_counts['pos']}/{self._evt_counts['drag']}/{self._evt_counts['press']}/{self._evt_counts['release']}/{self._evt_counts['scroll']}")
        lep = self._last_event_pos
        if all(np.isfinite(v) for v in lep):
            imgui.text(f"evt_mouse={int(lep[0])},{int(lep[1])}")
        else:
            imgui.text("evt_mouse=NA,NA")
        try:
            mp = tuple(io_dbg.mouse_pos)
            if len(mp) == 2 and all(np.isfinite(v) for v in mp):
                mp_disp = f"{int(mp[0])},{int(mp[1])}"
            else:
                mp_disp = "NA,NA"
        except Exception:
            mp_disp = "NA,NA"
        imgui.text(f"mouse_pos={mp_disp}")
        imgui.text(f"mouse_down={tuple(int(x) for x in io_dbg.mouse_down[:3])}")
        imgui.text(f"display_size={tuple(map(int, io_dbg.display_size))}")
        try:
            ms = getattr(self.wnd, 'mouse_states', None)
            if ms:
                imgui.text(f"wnd L/R/M: {int(ms.left)}/{int(ms.right)}/{int(ms.middle)}")
            # GL info
            try:
                info = getattr(self.ctx, 'info', {}) or {}
                gl_vendor = info.get('GL_VENDOR', 'unknown')
                gl_renderer = info.get('GL_RENDERER', 'unknown')
                imgui.text(f"GL_VENDOR: {gl_vendor}")
                imgui.text(f"GL_RENDERER: {gl_renderer}")
            except Exception:
                pass
            mx, my = self.wnd.mouse_position
            if np.isfinite(mx) and np.isfinite(my):
                imgui.text(f"wnd_mouse={int(mx)},{int(my)}")
            else:
                imgui.text("wnd_mouse=NA,NA")
            # Show last delta
            dxdy = self._last_evt_dxdy
            imgui.text(f"last dx,dy = {int(dxdy[0])},{int(dxdy[1])}")
            changed_rel, new_rel = imgui.checkbox("Relative mouse (lock cursor)", self.relative_mouse)
            if changed_rel:
                self.relative_mouse = new_rel
                try:
                    self.wnd.mouse_exclusivity = bool(new_rel)
                except Exception:
                    pass
            imgui.text(f"backend={type(self.wnd).__module__}")
        except Exception:
            pass
        hs_overlay = self.renderer.height_scale
        ch_hs2, hs_overlay = imgui.slider_float("Z exaggeration", hs_overlay, 0.01, 5.0, '%.2f')
        if ch_hs2:
            self.renderer.height_scale = float(hs_overlay)
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

        # Fallback per-frame camera control if motion events are not delivered
        io = imgui.get_io()
        if not io.want_capture_mouse and ((self._evt_counts['pos'] + self._evt_counts['drag']) == 0 or self.relative_mouse):
            try:
                raw_win = self._get_glfw_window()
                if raw_win is not None:
                    cx, cy = glfw.get_cursor_pos(raw_win)
                    if all(np.isfinite(v) for v in (cx, cy)) and all(np.isfinite(v) for v in self._last_mouse_pos):
                        dx = float(cx - self._last_mouse_pos[0])
                        dy = float(cy - self._last_mouse_pos[1])
                        self._last_mouse_pos = (float(cx), float(cy))
                        self._apply_camera_drag(dx, dy)
            except Exception:
                pass

    # Backwards-compatible name if the runtime expects "render"
    def render(self, time: float, frame_time: float):
        return self.on_render(time, frame_time)


def run_with_mglw():
    mglw.run_window_config(TerrainApp)


if __name__ == "__main__":
    run_with_mglw()
