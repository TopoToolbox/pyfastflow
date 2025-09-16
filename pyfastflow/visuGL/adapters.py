from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import moderngl

from .scene import Layer


class Heightfield3D(Layer):
    """Self-contained heightfield renderer using a trusted shader layout.

    Generates a texcoord-only mesh matching the heightmap aspect and renders
    from a DataHub-provided texture or a direct numpy array.
    """

    def __init__(self, mesh_max_dim: int = 2048):
        self.mesh_max_dim = int(mesh_max_dim)
        self._hub_name: Optional[str] = None
        self._hub = None
        self._height_scale = 0.5
        self._light = np.array([0.5, 1.0, 0.5], dtype=np.float32)
        self._sphere = False
        self._sphere_radius = 1.5
        self._local_arr: Optional[np.ndarray] = None

        # GL handles / state
        self._ctx = None
        self._program = None
        self._vao = None
        self._vbo_texcoords = None
        self._ibo = None
        self._tex = None
        self._cpu = None
        self.hm_nx = 512
        self.hm_ny = 512
        self.mesh_size = 4.0
        self.mesh_size_x = self.mesh_size
        self.mesh_size_z = self.mesh_size

    # Wiring ---------------------------------------------------------------
    def use_hub(self, tex_name: str) -> None:
        self._hub_name = tex_name

    def set_height_scale(self, v: float) -> None:
        self._height_scale = float(v)

    def set_light(self, dir3: Tuple[float, float, float] | np.ndarray) -> None:
        v = np.array(dir3, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        self._light = v.astype(np.float32)

    def set_sphere_mode(self, flag: bool) -> None:
        self._sphere = bool(flag)

    def set_heightmap(self, np_array: np.ndarray) -> None:
        self._local_arr = np.asarray(np_array)

    def set_mesh_max_dim(self, v: int) -> None:
        self.mesh_max_dim = int(max(2, v))
        if self._ctx is not None and self._program is not None:
            self._generate_mesh()

    # Layer interface ------------------------------------------------------
    def setup(self, ctx, hub) -> None:
        self._ctx = ctx
        self._hub = hub
        # Reasonable defaults for 3D terrain
        try:
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.disable(moderngl.CULL_FACE)
        except Exception:
            pass
        self._create_program()
        # Bind texture source
        if self._hub_name is not None:
            ent = hub._tex2d.get(self._hub_name)
            if ent and ent.get("tex") is not None:
                self._tex = ent["tex"]
                self.hm_nx = int(ent["shape"][1])
                self.hm_ny = int(ent["shape"][0])
                self._cpu = ent.get("cpu")
        elif self._local_arr is not None:
            name = f"_height_local_{id(self)}"
            hub.tex2d(name, self._local_arr.astype(np.float32, copy=False), fmt="R32F")
            ent = hub._tex2d.get(name)
            if ent:
                self._tex = ent["tex"]
                self.hm_nx = int(ent["shape"][1])
                self.hm_ny = int(ent["shape"][0])
                self._cpu = ent.get("cpu")
        self._generate_mesh()

    def draw(self, ctx, camera) -> None:
        # Refresh from hub and handle shape change
        if self._hub_name is not None and self._hub is not None:
            ent = self._hub._tex2d.get(self._hub_name)
            if ent and ent.get("tex") is not None:
                self._tex = ent["tex"]
                self._cpu = ent.get("cpu")
                w, h = self._tex.size
                if int(self.hm_nx) != int(w) or int(self.hm_ny) != int(h):
                    self.hm_nx = int(w)
                    self.hm_ny = int(h)
                    self._generate_mesh()

        if self._tex is None or self._vao is None or self._program is None:
            return

        # Matrices
        try:
            _, _, vw, vh = ctx.fbo.viewport
            aspect = float(max(1, vw)) / float(max(1, vh))
        except Exception:
            aspect = 16.0 / 9.0
        fov = np.radians(60.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        f = 1.0 / np.tan(fov / 2.0)
        proj[0, 0] = f / max(1e-6, aspect)
        proj[1, 1] = f
        proj[2, 2] = (100.0 + 0.1) / (0.1 - 100.0)
        proj[2, 3] = (2.0 * 100.0 * 0.1) / (0.1 - 100.0)
        proj[3, 2] = -1.0
        view = camera.get_view_matrix()

        # Bind and set uniforms
        self._tex.use(0)
        try:
            self._program['u_model'].write(np.eye(4, dtype=np.float32).T.tobytes())
            self._program['u_view'].write(view.T.astype('f4').tobytes())
            self._program['u_projection'].write(proj.T.astype('f4').tobytes())
            self._program['u_heightmap'] = 0
            self._program['u_height_scale'] = float(self._height_scale)
            self._program['u_mesh_size_x'] = float(self.mesh_size_x)
            self._program['u_mesh_size_z'] = float(self.mesh_size_z)
            self._program['u_sphere_mode'] = 1.0 if self._sphere else 0.0
            self._program['u_sphere_radius'] = float(self._sphere_radius)
            self._program['u_light_direction'].write(self._light.astype('f4').tobytes())
            self._program['u_min_height'] = 0.0
            self._program['u_max_height'] = float(self._height_scale)
            try:
                self._program['u_hide_underside'] = 1.0
            except Exception:
                pass
        except Exception:
            pass

        self._vao.render()

    # --------- CPU sampling + raycast (flat mode) ------------------------
    def _world_to_grid(self, x: float, z: float):
        half_x = float(self.mesh_size_x) * 0.5
        half_z = float(self.mesh_size_z) * 0.5
        if x < -half_x or x > half_x or z < -half_z or z > half_z:
            return None
        u = (x / (2.0 * half_x) + 0.5) * (self.hm_nx - 1)
        v = (z / (2.0 * half_z) + 0.5) * (self.hm_ny - 1)
        return u, v

    def _sample_height(self, x: float, z: float) -> Optional[float]:
        grid = self._cpu
        if grid is None:
            return None
        uv = self._world_to_grid(x, z)
        if uv is None:
            return None
        u, v = uv
        i0 = int(np.floor(u))
        j0 = int(np.floor(v))
        i1 = min(i0 + 1, self.hm_nx - 1)
        j1 = min(j0 + 1, self.hm_ny - 1)
        du = u - i0
        dv = v - j0
        h00 = float(grid[j0, i0])
        h10 = float(grid[j0, i1])
        h01 = float(grid[j1, i0])
        h11 = float(grid[j1, i1])
        h0 = h00 * (1 - du) + h10 * du
        h1 = h01 * (1 - du) + h11 * du
        h = h0 * (1 - dv) + h1 * dv
        return h * float(self._height_scale)

    def intersect_ray(self, ray_origin: np.ndarray, ray_dir: np.ndarray):
        rd = np.array(ray_dir, dtype=np.float32)
        n = float(np.linalg.norm(rd))
        if n == 0.0:
            return None
        rd /= n
        ro = np.array(ray_origin, dtype=np.float32)
        if self._sphere:
            # Approximate: intersect base sphere only as fallback
            R = float(self._sphere_radius)
            o = ro
            d = rd
            b = 2.0 * np.dot(o, d)
            c = np.dot(o, o) - R * R
            disc = b * b - 4.0 * c
            if disc < 0.0:
                return None
            t = (-b - np.sqrt(disc)) / 2.0
            if t <= 0.0:
                t = (-b + np.sqrt(disc)) / 2.0
            if t <= 0.0:
                return None
            return o + d * t
        # Flat mode: ray-march for y = h(x,z)
        t = 0.0
        t_max = float(self.mesh_size) * 6.0
        dt = max(self.mesh_size / max(2, max(self.hm_nx, self.hm_ny)), 0.002)
        prev_s = None
        prev_t = None
        while t <= t_max:
            p = ro + rd * t
            h = self._sample_height(float(p[0]), float(p[2]))
            s = p[1] - (h if h is not None else p[1] + 1.0)
            if prev_s is not None and prev_s > 0.0 and s <= 0.0:
                a = prev_t
                b = t
                for _ in range(12):
                    m = 0.5 * (a + b)
                    pm = ro + rd * m
                    hm = self._sample_height(float(pm[0]), float(pm[2]))
                    sm = pm[1] - (hm if hm is not None else pm[1] + 1.0)
                    if sm > 0.0:
                        a = m
                    else:
                        b = m
                return ro + rd * (0.5 * (a + b))
            prev_s = s
            prev_t = t
            t += dt * (1.0 + 0.5 * t / (self.mesh_size + 1e-6))
        return None

    # Internals ------------------------------------------------------------
    def _create_program(self) -> None:
        vs = """
        #version 330 core
        in vec2 in_texcoord;
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform sampler2D u_heightmap;
        uniform float u_height_scale;
        uniform float u_sphere_mode; // >0.5 => sphere mapping
        uniform float u_sphere_radius;
        uniform float u_mesh_size_x;
        uniform float u_mesh_size_z;
        out vec3 v_world_pos;
        out vec3 v_normal;
        out float v_height;
        const float PI = 3.14159265358979323846;
        vec3 pos_from_uv(vec2 uv) {
            float h = texture(u_heightmap, uv).r * u_height_scale;
            if (u_sphere_mode > 0.5) {
                float lon = (uv.x - 0.5) * 2.0 * PI;
                float lat = (uv.y - 0.5) * PI;
                float cl = cos(lat);
                vec3 dir = vec3(cl * cos(lon), sin(lat), cl * sin(lon));
                float r = u_sphere_radius + h;
                return dir * r;
            } else {
                float x = (uv.x - 0.5) * u_mesh_size_x;
                float z = (uv.y - 0.5) * u_mesh_size_z;
                return vec3(x, h, z);
            }
        }
        void main() {
            vec2 texel_size = 1.0 / textureSize(u_heightmap, 0);
            vec3 p_c = pos_from_uv(in_texcoord);
            vec3 p_r = pos_from_uv(in_texcoord + vec2(texel_size.x, 0.0));
            vec3 p_l = pos_from_uv(in_texcoord - vec2(texel_size.x, 0.0));
            vec3 p_u = pos_from_uv(in_texcoord + vec2(0.0, texel_size.y));
            vec3 p_d = pos_from_uv(in_texcoord - vec2(0.0, texel_size.y));
            vec3 n = normalize(cross(p_u - p_d, p_r - p_l));
            v_world_pos = (u_model * vec4(p_c, 1.0)).xyz;
            v_normal = normalize((u_model * vec4(n, 0.0)).xyz);
            v_height = texture(u_heightmap, in_texcoord).r * u_height_scale;
            gl_Position = u_projection * u_view * vec4(v_world_pos, 1.0);
        }
        """
        fs = """
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
        self._program = self._ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _generate_mesh(self) -> None:
        # Choose mesh resolution based on heightmap aspect
        hm_aspect = float(self.hm_nx) / max(1.0, float(self.hm_ny))
        if hm_aspect >= 1.0:
            nx = int(self.mesh_max_dim)
            ny = max(2, int(round(self.mesh_max_dim / max(1e-6, hm_aspect))))
        else:
            ny = int(self.mesh_max_dim)
            nx = max(2, int(round(self.mesh_max_dim * max(1e-6, hm_aspect))))

        # World extents keep aspect
        if hm_aspect >= 1.0:
            self.mesh_size_x = float(self.mesh_size)
            self.mesh_size_z = float(self.mesh_size) / max(1e-6, hm_aspect)
        else:
            self.mesh_size_z = float(self.mesh_size)
            self.mesh_size_x = float(self.mesh_size) * max(1e-6, hm_aspect)

        u = np.linspace(0.0, 1.0, nx, dtype=np.float32)
        v = np.linspace(0.0, 1.0, ny, dtype=np.float32)
        U, V = np.meshgrid(u, v, indexing='xy')
        texcoords = np.stack([U, V], axis=-1).reshape(-1, 2).astype(np.float32, copy=False)
        ii = np.arange(nx - 1, dtype=np.uint32)
        jj = np.arange(ny - 1, dtype=np.uint32)
        II, JJ = np.meshgrid(ii, jj, indexing='xy')
        tl = JJ * nx + II
        tr = tl + 1
        bl = tl + nx
        br = bl + 1
        tri1 = np.stack([tl, bl, tr], axis=-1)
        tri2 = np.stack([tr, bl, br], axis=-1)
        indices = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).astype(np.uint32, copy=False).reshape(-1)

        if self._vbo_texcoords is not None:
            try:
                self._vbo_texcoords.release()
            except Exception:
                pass
        if self._ibo is not None:
            try:
                self._ibo.release()
            except Exception:
                pass
        self._vbo_texcoords = self._ctx.buffer(np.ascontiguousarray(texcoords))
        self._ibo = self._ctx.buffer(np.ascontiguousarray(indices))
        self._vao = self._ctx.vertex_array(self._program, [(self._vbo_texcoords, '2f', 'in_texcoord')], self._ibo)


class Image2D(Layer):
    """Minimal 2D image view (screen-aligned)."""

    def __init__(self):
        self._hub_name: Optional[str] = None
        self._program = None
        self._vao = None
        self._tex = None

    def use_hub(self, tex_name: str) -> None:
        self._hub_name = tex_name

    def setup(self, ctx, hub) -> None:
        vs = """
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main(){
            v_uv = in_uv;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        fs = """
        #version 330 core
        uniform sampler2D u_tex;
        in vec2 v_uv;
        out vec4 frag;
        void main(){
            frag = texture(u_tex, v_uv);
        }
        """
        self._program = ctx.program(vertex_shader=vs, fragment_shader=fs)
        # Fullscreen quad
        quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,   1.0, 1.0, 1.0,
        ], dtype=np.float32)
        idx = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
        vbo = ctx.buffer(quad.tobytes())
        ibo = ctx.buffer(idx.tobytes())
        self._vao = ctx.vertex_array(self._program, [(vbo, '2f 2f', 'in_pos', 'in_uv')], ibo)
        if self._hub_name is not None:
            ent = hub._tex2d.get(self._hub_name)
            if ent:
                self._tex = ent["tex"]

    def draw(self, ctx, camera) -> None:
        if self._tex is None:
            return
        self._tex.use(0)
        try:
            self._program['u_tex'] = 0
        except Exception:
            pass
        self._vao.render()
