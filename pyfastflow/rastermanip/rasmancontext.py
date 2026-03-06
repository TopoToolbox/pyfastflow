"""
Raster manipulation context.

This context is intentionally dimension-explicit: kernels are not bound to one
GridContext and can process many raster sizes.

Author: B.G (02/2026)
"""

from types import FunctionType, SimpleNamespace

import numpy as np
import taichi as ti

from .. import constants as cte
from .. import pool as ppool
from ..grid import GridContext
from . import rasman_kernels as rk


class RasManContext:
    """
    Raster manipulation context with explicit-dimension kernels and wrappers.

    Main APIs:
    - ``upscale_grid``: nearest / bilinear / bicubic / lanczos
    - ``downscale_grid``: mean / median / min / max / percentile

    Other methods are convenience wrappers around these two.

    Author: B.G (02/2026)
    """

    def __init__(self):
        """
        Build helper and kernel namespaces.

        Author: B.G (02/2026)
        """
        self.tfunc = SimpleNamespace()
        self.kernels = SimpleNamespace()
        self._compile_helpers()
        self._compile_kernels()

    def make_kernel(self, kernel_template, **extra_globals):
        """
        Specialize one generic kernel against this context.

        Author: B.G (02/2026)
        """
        return self._specialize_callable(kernel_template, ti.kernel, **extra_globals)

    def make_func(self, func_template, **extra_globals):
        """
        Specialize one generic helper against this context.

        Author: B.G (02/2026)
        """
        return self._specialize_callable(func_template, ti.func, **extra_globals)

    def _specialize_callable(self, func_template, decorator, **extra_globals):
        """
        Clone and inject this context plus helper overrides.

        Author: B.G (02/2026)
        """
        source = getattr(func_template, "__wrapped__", func_template)
        func_globals = dict(source.__globals__)
        func_globals["rasmanctx"] = self
        func_globals.update(extra_globals)

        specialised = FunctionType(
            source.__code__,
            func_globals,
            source.__name__,
            source.__defaults__,
            source.__closure__,
        )
        specialised.__kwdefaults__ = source.__kwdefaults__
        specialised.__annotations__ = dict(source.__annotations__)
        specialised.__doc__ = source.__doc__
        specialised.__qualname__ = source.__qualname__
        return decorator(specialised)

    def _compile_helpers(self):
        """
        Bind helper Taichi functions to ``tfunc``.

        Author: B.G (02/2026)
        """
        self.tfunc.wrap_index = self.make_func(rk.wrap_index)
        self.tfunc.reflect_index = self.make_func(rk.reflect_index)
        self.tfunc.resolve_index = self.make_func(
            rk.resolve_index,
            wrap_index=self.tfunc.wrap_index,
            reflect_index=self.tfunc.reflect_index,
        )
        self.tfunc.cubic_interpolate = self.make_func(rk.cubic_interpolate)
        self.tfunc.sinc = self.make_func(rk.sinc)
        self.tfunc.lanczos_weight = self.make_func(rk.lanczos_weight, sinc=self.tfunc.sinc)

        self.tfunc.sample_nearest = self.make_func(
            rk.sample_nearest,
            resolve_index=self.tfunc.resolve_index,
        )
        self.tfunc.sample_bilinear = self.make_func(
            rk.sample_bilinear,
            resolve_index=self.tfunc.resolve_index,
        )
        self.tfunc.sample_bicubic = self.make_func(
            rk.sample_bicubic,
            resolve_index=self.tfunc.resolve_index,
            cubic_interpolate=self.tfunc.cubic_interpolate,
        )
        self.tfunc.sample_lanczos8 = self.make_func(
            rk.sample_lanczos8,
            resolve_index=self.tfunc.resolve_index,
            lanczos_weight=self.tfunc.lanczos_weight,
        )
        self.tfunc.source_coord_1d = self.make_func(rk.source_coord_1d)
        self.tfunc.box_bounds_1d = self.make_func(rk.box_bounds_1d)

    def _compile_kernels(self):
        """
        Bind raw kernels to ``kernels``.

        Author: B.G (02/2026)
        """
        self.kernels.two_d_to_flat = self.make_kernel(rk.two_d_to_flat_kernel)
        self.kernels.flat_to_2d = self.make_kernel(rk.flat_to_2d_kernel)

        self.kernels.upscale_nearest = self.make_kernel(
            rk.upscale_nearest_kernel,
            source_coord_1d=self.tfunc.source_coord_1d,
            sample_nearest=self.tfunc.sample_nearest,
        )
        self.kernels.upscale_bilinear = self.make_kernel(
            rk.upscale_bilinear_kernel,
            source_coord_1d=self.tfunc.source_coord_1d,
            sample_bilinear=self.tfunc.sample_bilinear,
        )
        self.kernels.upscale_bicubic = self.make_kernel(
            rk.upscale_bicubic_kernel,
            source_coord_1d=self.tfunc.source_coord_1d,
            sample_bicubic=self.tfunc.sample_bicubic,
        )
        self.kernels.upscale_lanczos = self.make_kernel(
            rk.upscale_lanczos8_kernel,
            source_coord_1d=self.tfunc.source_coord_1d,
            sample_lanczos8=self.tfunc.sample_lanczos8,
        )

        self.kernels.downscale_mean = self.make_kernel(
            rk.downscale_mean_kernel,
            box_bounds_1d=self.tfunc.box_bounds_1d,
        )
        self.kernels.downscale_min = self.make_kernel(
            rk.downscale_min_kernel,
            box_bounds_1d=self.tfunc.box_bounds_1d,
        )
        self.kernels.downscale_max = self.make_kernel(
            rk.downscale_max_kernel,
            box_bounds_1d=self.tfunc.box_bounds_1d,
        )
        self.kernels.downscale_median = self.make_kernel(
            rk.downscale_median_kernel,
            box_bounds_1d=self.tfunc.box_bounds_1d,
        )
        self.kernels.downscale_percentile = self.make_kernel(
            rk.downscale_percentile_kernel,
            box_bounds_1d=self.tfunc.box_bounds_1d,
        )

    def _unwrap_field(self, data):
        """
        Return Taichi field from raw field or TPField.

        Author: B.G (02/2026)
        """
        return data.field if hasattr(data, "field") else data

    def _normalize_output_layout(self, output_layout: str) -> str:
        """
        Validate output layout.

        Author: B.G (02/2026)
        """
        layout = str(output_layout).lower()
        if layout not in {"flat", "2d"}:
            raise ValueError("output_layout must be 'flat' or '2d'")
        return layout

    def _normalize_boundary(self, boundary: str) -> int:
        """
        Convert boundary string to integer code.

        Author: B.G (02/2026)
        """
        b = str(boundary).lower()
        table = {
            "clamp": rk.BOUNDARY_CLAMP,
            "wrap": rk.BOUNDARY_WRAP,
            "reflect": rk.BOUNDARY_REFLECT,
        }
        if b not in table:
            raise ValueError("boundary must be 'clamp', 'wrap', or 'reflect'")
        return table[b]

    def _normalize_upscale_method(self, method: str) -> str:
        """
        Validate interpolation method for upscaling/resampling.

        Author: B.G (02/2026)
        """
        m = str(method).lower()
        if m not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("upscale method must be one of: nearest, bilinear, bicubic, lanczos")
        return m

    def _normalize_downscale_method(self, method: str) -> str:
        """
        Validate reduction method for downscaling.

        Author: B.G (02/2026)
        """
        m = str(method).lower()
        if m not in {"mean", "median", "min", "max", "percentile"}:
            raise ValueError("downscale method must be one of: mean, median, min, max, percentile")
        return m

    def _prepare_source_flat(self, grid_data, nx=None, ny=None):
        """
        Convert input numpy/field data into one flat pooled field.

        Author: B.G (02/2026)
        """
        src = self._unwrap_field(grid_data)

        if isinstance(src, np.ndarray):
            if src.ndim != 2:
                raise ValueError("Input numpy array must be 2D")
            ny_i, nx_i = src.shape
            out = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (nx_i * ny_i))
            out.from_numpy(np.asarray(src, dtype=np.float32).reshape(-1))
            return out, int(nx_i), int(ny_i)

        if hasattr(src, "shape") and hasattr(src, "to_numpy"):
            shape = tuple(src.shape)
            if len(shape) == 1:
                if nx is None or ny is None:
                    raise ValueError("nx and ny must be provided for 1D Taichi fields")
                if int(nx) * int(ny) != shape[0]:
                    raise ValueError("nx * ny does not match 1D field size")
                out = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (shape[0],))
                out.field.copy_from(src)
                return out, int(nx), int(ny)
            if len(shape) == 2:
                ny_i, nx_i = shape
                out = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (nx_i * ny_i))
                self.kernels.two_d_to_flat(src, out.field, int(nx_i), int(ny_i))
                return out, int(nx_i), int(ny_i)
            raise ValueError("Input Taichi field must be 1D or 2D")

        raise TypeError("grid_data must be a 2D numpy array, Taichi field, or TPField")

    def _finalize_output(self, target_flat, nx_t: int, ny_t: int, output_layout: str, as_numpy: bool):
        """
        Convert flat output to requested layout/type.

        Author: B.G (02/2026)
        """
        if as_numpy:
            arr = target_flat.field.to_numpy()
            target_flat.release()
            return arr.reshape(-1) if output_layout == "flat" else arr.reshape((ny_t, nx_t))

        if output_layout == "flat":
            return target_flat

        out_2d = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (ny_t, nx_t))
        self.kernels.flat_to_2d(target_flat.field, out_2d.field, nx_t, ny_t)
        target_flat.release()
        return out_2d

    def _run_upscale_kernel(
        self,
        source_flat,
        target_flat,
        nx_src: int,
        ny_src: int,
        nx_t: int,
        ny_t: int,
        method: str,
        boundary_mode: int,
    ):
        """
        Dispatch one interpolation kernel.

        Author: B.G (02/2026)
        """
        if method == "nearest":
            self.kernels.upscale_nearest(source_flat, target_flat, nx_src, ny_src, nx_t, ny_t, boundary_mode)
        elif method == "bilinear":
            self.kernels.upscale_bilinear(source_flat, target_flat, nx_src, ny_src, nx_t, ny_t, boundary_mode)
        elif method == "bicubic":
            self.kernels.upscale_bicubic(source_flat, target_flat, nx_src, ny_src, nx_t, ny_t, boundary_mode)
        else:
            self.kernels.upscale_lanczos(source_flat, target_flat, nx_src, ny_src, nx_t, ny_t, boundary_mode)

    def _make_output_gridctx(
        self,
        nx_in: int,
        ny_in: int,
        nx_out: int,
        ny_out: int,
        return_gridctx: bool,
        gridctx=None,
        dx: float | None = None,
        dx_out: float | None = None,
        topology: str | None = None,
        boundary_mode: str | None = None,
        has_bcs: bool = False,
    ):
        """
        Optionally create the output GridContext.

        Author: B.G (02/2026)
        """
        if not return_gridctx:
            return None

        if dx_out is None:
            base_dx = dx if dx is not None else (gridctx.dx if gridctx is not None else None)
            if base_dx is None:
                raise ValueError("return_gridctx=True requires dx_out, dx, or gridctx")

            sx = nx_out / float(nx_in)
            sy = ny_out / float(ny_in)
            if abs(sx - sy) > 1e-12:
                raise ValueError("Anisotropic resize requires explicit dx_out")
            dx_out = base_dx / sx

        topo = topology if topology is not None else (gridctx.topology if gridctx is not None else "D4")
        bmode = boundary_mode if boundary_mode is not None else (
            gridctx.boundary_mode if gridctx is not None else "normal"
        )

        return GridContext(
            nx=nx_out,
            ny=ny_out,
            dx=float(dx_out),
            boundary_mode=bmode,
            topology=topo,
            has_bcs=bool(has_bcs),
        )

    def upscale_grid(
        self,
        grid_data,
        scale_factor: float | None = None,
        target_nx: int | None = None,
        target_ny: int | None = None,
        method: str = "bicubic",
        boundary: str = "clamp",
        output_layout: str = "2d",
        as_numpy: bool = False,
        nx: int | None = None,
        ny: int | None = None,
        return_gridctx: bool = False,
        gridctx=None,
        dx: float | None = None,
        dx_out: float | None = None,
        topology: str | None = None,
        boundary_mode: str | None = None,
        has_bcs: bool = False,
    ):
        """
        Upscale/resample a grid with interpolation methods.

        Methods: nearest, bilinear, bicubic, lanczos.

        Author: B.G (02/2026)
        """
        method = self._normalize_upscale_method(method)
        output_layout = self._normalize_output_layout(output_layout)
        bmode = self._normalize_boundary(boundary)

        source, nx_i, ny_i = self._prepare_source_flat(grid_data, nx=nx, ny=ny)
        if target_nx is None or target_ny is None:
            if scale_factor is None:
                source.release()
                raise ValueError("Provide either target_nx/target_ny or scale_factor")
            if scale_factor <= 0.0:
                source.release()
                raise ValueError("scale_factor must be > 0")
            nx_o = max(1, int(round(nx_i * float(scale_factor))))
            ny_o = max(1, int(round(ny_i * float(scale_factor))))
        else:
            nx_o = int(target_nx)
            ny_o = int(target_ny)
            if nx_o <= 0 or ny_o <= 0:
                source.release()
                raise ValueError("target_nx and target_ny must be > 0")

        target = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (nx_o * ny_o))
        try:
            self._run_upscale_kernel(
                source.field,
                target.field,
                nx_i,
                ny_i,
                nx_o,
                ny_o,
                method,
                bmode,
            )
        finally:
            source.release()

        result = self._finalize_output(target, nx_o, ny_o, output_layout, as_numpy)
        out_gridctx = self._make_output_gridctx(
            nx_i,
            ny_i,
            nx_o,
            ny_o,
            return_gridctx=return_gridctx,
            gridctx=gridctx,
            dx=dx,
            dx_out=dx_out,
            topology=topology,
            boundary_mode=boundary_mode,
            has_bcs=has_bcs,
        )
        if return_gridctx:
            return result, out_gridctx
        return result

    def downscale_grid(
        self,
        grid_data,
        scale_factor: float | None = None,
        target_nx: int | None = None,
        target_ny: int | None = None,
        method: str = "mean",
        percentile: float = 50.0,
        percentile_iterations: int = 20,
        pre_decimate: bool = False,
        pre_decimate_factor: int = 4,
        decimation_method: str = "nearest",
        boundary: str = "clamp",
        output_layout: str = "2d",
        as_numpy: bool = False,
        nx: int | None = None,
        ny: int | None = None,
        return_gridctx: bool = False,
        gridctx=None,
        dx: float | None = None,
        dx_out: float | None = None,
        topology: str | None = None,
        boundary_mode: str | None = None,
        has_bcs: bool = False,
    ):
        """
        Downscale a grid with box-statistics reducers.

        Methods: mean, median, min, max, percentile.
        Optional pre-decimation can reduce compute for very large downscales.

        Author: B.G (02/2026)
        """
        method = self._normalize_downscale_method(method)
        output_layout = self._normalize_output_layout(output_layout)
        bmode = self._normalize_boundary(boundary)

        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("percentile must be in [0, 100]")

        source, nx_i, ny_i = self._prepare_source_flat(grid_data, nx=nx, ny=ny)
        if target_nx is None or target_ny is None:
            if scale_factor is None:
                source.release()
                raise ValueError("Provide either target_nx/target_ny or scale_factor")
            if scale_factor <= 0.0:
                source.release()
                raise ValueError("scale_factor must be > 0")
            nx_o = max(1, int(round(nx_i * float(scale_factor))))
            ny_o = max(1, int(round(ny_i * float(scale_factor))))
        else:
            nx_o = int(target_nx)
            ny_o = int(target_ny)
            if nx_o <= 0 or ny_o <= 0:
                source.release()
                raise ValueError("target_nx and target_ny must be > 0")

        work = source
        work_nx = nx_i
        work_ny = ny_i

        try:
            if pre_decimate:
                m = self._normalize_upscale_method(decimation_method)
                fac = max(1, int(pre_decimate_factor))
                nx_mid = min(work_nx, max(nx_o * fac, nx_o))
                ny_mid = min(work_ny, max(ny_o * fac, ny_o))

                if nx_mid < work_nx or ny_mid < work_ny:
                    pre = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (nx_mid * ny_mid))
                    self._run_upscale_kernel(
                        work.field,
                        pre.field,
                        work_nx,
                        work_ny,
                        nx_mid,
                        ny_mid,
                        m,
                        bmode,
                    )
                    if work is not source:
                        work.release()
                    work = pre
                    work_nx = nx_mid
                    work_ny = ny_mid

            target = ppool.taipool.get_tpfield(cte.FLOAT_TYPE_TI, (nx_o * ny_o))
            if method == "mean":
                self.kernels.downscale_mean(work.field, target.field, work_nx, work_ny, nx_o, ny_o)
            elif method == "min":
                self.kernels.downscale_min(work.field, target.field, work_nx, work_ny, nx_o, ny_o)
            elif method == "max":
                self.kernels.downscale_max(work.field, target.field, work_nx, work_ny, nx_o, ny_o)
            elif method == "median":
                self.kernels.downscale_median(
                    work.field,
                    target.field,
                    work_nx,
                    work_ny,
                    nx_o,
                    ny_o,
                    int(percentile_iterations),
                )
            else:
                pctl = float(percentile)
                self.kernels.downscale_percentile(
                    work.field,
                    target.field,
                    work_nx,
                    work_ny,
                    nx_o,
                    ny_o,
                    pctl,
                    int(percentile_iterations),
                )
        finally:
            if work is not source:
                work.release()
            source.release()

        result = self._finalize_output(target, nx_o, ny_o, output_layout, as_numpy)
        out_gridctx = self._make_output_gridctx(
            nx_i,
            ny_i,
            nx_o,
            ny_o,
            return_gridctx=return_gridctx,
            gridctx=gridctx,
            dx=dx,
            dx_out=dx_out,
            topology=topology,
            boundary_mode=boundary_mode,
            has_bcs=has_bcs,
        )
        if return_gridctx:
            return result, out_gridctx
        return result

    def double_resolution(self, grid_data, method: str = "bicubic", **kwargs):
        """
        Convenience wrapper around ``upscale_grid(..., scale_factor=2)``.

        Author: B.G (02/2026)
        """
        return self.upscale_grid(
            grid_data=grid_data,
            scale_factor=2.0,
            method=method,
            **kwargs,
        )

    def halve_resolution(self, grid_data, method: str = "mean", **kwargs):
        """
        Convenience wrapper around ``downscale_grid(..., scale_factor=0.5)``.

        Author: B.G (02/2026)
        """
        return self.downscale_grid(
            grid_data=grid_data,
            scale_factor=0.5,
            method=method,
            **kwargs,
        )

    def resize_raster(
        self,
        grid_data,
        scale_factor: float,
        upscale_method: str = "bicubic",
        downscale_method: str = "mean",
        **kwargs,
    ):
        """
        Wrapper dispatching to ``upscale_grid`` or ``downscale_grid``.

        Author: B.G (02/2026)
        """
        if scale_factor <= 0.0:
            raise ValueError("scale_factor must be > 0")
        if scale_factor >= 1.0:
            return self.upscale_grid(
                grid_data=grid_data,
                scale_factor=scale_factor,
                method=upscale_method,
                **kwargs,
            )
        return self.downscale_grid(
            grid_data=grid_data,
            scale_factor=scale_factor,
            method=downscale_method,
            **kwargs,
        )

    def resize_to_dims(
        self,
        grid_data,
        target_nx: int,
        target_ny: int,
        upscale_method: str = "bicubic",
        downscale_method: str = "mean",
        mixed_method: str = "bilinear",
        nx: int | None = None,
        ny: int | None = None,
        **kwargs,
    ):
        """
        Resize to explicit dimensions.

        If both dimensions grow -> ``upscale_grid``.
        If both shrink -> ``downscale_grid``.
        If mixed (one up, one down) -> interpolation path with ``mixed_method``.

        Author: B.G (02/2026)
        """
        source, nx_i, ny_i = self._prepare_source_flat(grid_data, nx=nx, ny=ny)
        source.release()

        nx_t = int(target_nx)
        ny_t = int(target_ny)
        if nx_t <= 0 or ny_t <= 0:
            raise ValueError("target_nx and target_ny must be > 0")

        if nx_t >= nx_i and ny_t >= ny_i:
            return self.upscale_grid(
                grid_data=grid_data,
                target_nx=nx_t,
                target_ny=ny_t,
                method=upscale_method,
                nx=nx,
                ny=ny,
                **kwargs,
            )

        if nx_t <= nx_i and ny_t <= ny_i:
            return self.downscale_grid(
                grid_data=grid_data,
                target_nx=nx_t,
                target_ny=ny_t,
                method=downscale_method,
                nx=nx,
                ny=ny,
                **kwargs,
            )

        return self.upscale_grid(
            grid_data=grid_data,
            target_nx=nx_t,
            target_ny=ny_t,
            method=mixed_method,
            nx=nx,
            ny=ny,
            **kwargs,
        )

    def resize_to_max_dim(
        self,
        grid_data,
        max_dim: int,
        upscale_method: str = "bicubic",
        downscale_method: str = "mean",
        nx: int | None = None,
        ny: int | None = None,
        **kwargs,
    ):
        """
        Resize so that ``max(nx, ny) == max_dim`` unless already smaller.

        Author: B.G (02/2026)
        """
        if int(max_dim) <= 0:
            raise ValueError("max_dim must be > 0")

        source, nx_i, ny_i = self._prepare_source_flat(grid_data, nx=nx, ny=ny)
        source.release()

        cur_max = max(nx_i, ny_i)
        if cur_max <= int(max_dim):
            return self.resize_to_dims(
                grid_data=grid_data,
                target_nx=nx_i,
                target_ny=ny_i,
                upscale_method=upscale_method,
                downscale_method=downscale_method,
                nx=nx,
                ny=ny,
                **kwargs,
            )

        scale = float(max_dim) / float(cur_max)
        return self.resize_raster(
            grid_data=grid_data,
            scale_factor=scale,
            upscale_method=upscale_method,
            downscale_method=downscale_method,
            nx=nx,
            ny=ny,
            **kwargs,
        )
