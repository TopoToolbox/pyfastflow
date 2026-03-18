import taichi as ti

from .. import constants as cte
from ..context import ContextFactory, ContextRef
from ..flow import FlowContext
from ._lem_param_helpers import (
    get_K_bedrock,
    get_K_sed,
    get_domain,
    get_dt,
    get_kappa_bedrock,
    get_kappa_sed,
    get_m_exp,
    get_n_exp,
    get_uplift_rate,
)
from .lem_kernels import (
    assemble_hillslope_adi_row_system_kernel,
    build_hillslope_fixed_mask_kernel,
    flat_to_grid_2d_kernel,
    grid_2d_to_flat_kernel,
    init_erode_spl_kernel,
    iteration_erode_spl_kernel,
    solve_cyclic_rows_kernel,
    solve_tridiagonal_rows_kernel,
    tectonic_uplift_kernel,
    transpose_grid_2d_kernel,
    uplift_baselevel_kernel,
)


class LEMContext:
    """
    Flat landscape-evolution specialization context.

    This context only stores parameter state plus the compiled uplift,
    incision, and hillslope kernels specialized against one grid/flow setup.

    Author: B.G (03/2026)
    """

    def __init__(
        self,
        gridctx,
        flowctx=None,
        dt_mode: str = "const",
        dt: float = 1.0,
        uplift_rate_mode: str = "const",
        uplift_rate: float = 0.0,
        K_bedrock_mode: str = "const",
        K_bedrock: float = 0.0,
        m_exp_mode: str = "const",
        m_exp: float = 0.5,
        n_exp_mode: str = "const",
        n_exp: float = 1.0,
        K_sed_mode: str = "const",
        K_sed: float = 0.0,
        kappa_bedrock_mode: str = "const",
        kappa_bedrock: float = 0.0,
        kappa_sed_mode: str = "const",
        kappa_sed: float = 0.0,
        domain_mode: str = "const",
        domain: int = 1,
    ):
        self.gridctx = gridctx
        self.flowctx = flowctx if flowctx is not None else FlowContext(gridctx)
        self.gactx = self.flowctx.gactx
        self.n_flat = self.gridctx.n_flat
        self.logn = self.flowctx.logn

        self._factory = ContextFactory(
            self,
            bindings={
                "gridctx": self.gridctx,
                "flowctx": self.flowctx,
                "lemctx": self,
            },
            n_flat=self.n_flat,
        )

        float_params = [
            ("dt", dt_mode, dt),
            ("uplift_rate", uplift_rate_mode, uplift_rate),
            ("K_bedrock", K_bedrock_mode, K_bedrock),
            ("m_exp", m_exp_mode, m_exp),
            ("n_exp", n_exp_mode, n_exp),
            ("K_sed", K_sed_mode, K_sed),
            ("kappa_bedrock", kappa_bedrock_mode, kappa_bedrock),
            ("kappa_sed", kappa_sed_mode, kappa_sed),
        ]
        for name, mode, value in float_params:
            self._factory.params.declare(
                name,
                dtype=cte.FLOAT_TYPE_TI,
                mode=mode,
                value=value,
            )
            self._factory.params.bind_setter(name)

        self._factory.params.declare(
            "domain",
            dtype=ti.u8,
            mode=domain_mode,
            value=domain,
        )
        self._factory.params.bind_setter("domain")

        self._factory.compile_block(
            [
                {"target": "tfunc", "name": "get_dt", "template": get_dt, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "get_uplift_rate",
                    "template": get_uplift_rate,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_K_bedrock",
                    "template": get_K_bedrock,
                    "kind": "func",
                },
                {"target": "tfunc", "name": "get_m_exp", "template": get_m_exp, "kind": "func"},
                {"target": "tfunc", "name": "get_n_exp", "template": get_n_exp, "kind": "func"},
                {"target": "tfunc", "name": "get_K_sed", "template": get_K_sed, "kind": "func"},
                {
                    "target": "tfunc",
                    "name": "get_kappa_bedrock",
                    "template": get_kappa_bedrock,
                    "kind": "func",
                },
                {
                    "target": "tfunc",
                    "name": "get_kappa_sed",
                    "template": get_kappa_sed,
                    "kind": "func",
                },
                {"target": "tfunc", "name": "get_domain", "template": get_domain, "kind": "func"},
            ]
        )

        self._factory.compile_block(
            [
                {
                    "target": "kernels",
                    "name": "tectonic_uplift",
                    "template": tectonic_uplift_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_dt": ContextRef("tfunc.get_dt"),
                        "get_uplift_rate": ContextRef("tfunc.get_uplift_rate"),
                    },
                },
                {
                    "target": "kernels",
                    "name": "uplift_baselevel",
                    "template": uplift_baselevel_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_dt": ContextRef("tfunc.get_dt"),
                        "get_uplift_rate": ContextRef("tfunc.get_uplift_rate"),
                    },
                },
                {
                    "target": "kernels",
                    "name": "init_erode_spl",
                    "template": init_erode_spl_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_dt": ContextRef("tfunc.get_dt"),
                        "get_K_bedrock": ContextRef("tfunc.get_K_bedrock"),
                        "get_m_exp": ContextRef("tfunc.get_m_exp"),
                    },
                },
                {
                    "target": "kernels",
                    "name": "iteration_erode_spl",
                    "template": iteration_erode_spl_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.hillslope",
                    "name": "flat_to_grid",
                    "template": flat_to_grid_2d_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.hillslope",
                    "name": "grid_to_flat",
                    "template": grid_2d_to_flat_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.hillslope",
                    "name": "transpose",
                    "template": transpose_grid_2d_kernel,
                    "kind": "kernel",
                },
                {
                    "target": "kernels.hillslope",
                    "name": "build_fixed_mask",
                    "template": build_hillslope_fixed_mask_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": False},
                },
                {
                    "target": "kernels.hillslope",
                    "name": "build_fixed_mask_transposed",
                    "template": build_hillslope_fixed_mask_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": True},
                },
                {
                    "target": "kernels.hillslope",
                    "name": "assemble_rows",
                    "template": assemble_hillslope_adi_row_system_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_dt": ContextRef("tfunc.get_dt"),
                        "get_kappa_bedrock": ContextRef("tfunc.get_kappa_bedrock"),
                        "transposed": False,
                    },
                },
                {
                    "target": "kernels.hillslope",
                    "name": "assemble_rows_transposed",
                    "template": assemble_hillslope_adi_row_system_kernel,
                    "kind": "kernel",
                    "bindings": {
                        "get_dt": ContextRef("tfunc.get_dt"),
                        "get_kappa_bedrock": ContextRef("tfunc.get_kappa_bedrock"),
                        "transposed": True,
                    },
                },
                {
                    "target": "kernels.hillslope",
                    "name": "solve_rows",
                    "template": solve_tridiagonal_rows_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": False},
                },
                {
                    "target": "kernels.hillslope",
                    "name": "solve_rows_transposed",
                    "template": solve_tridiagonal_rows_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": True},
                },
                {
                    "target": "kernels.hillslope",
                    "name": "solve_rows_cyclic",
                    "template": solve_cyclic_rows_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": False},
                },
                {
                    "target": "kernels.hillslope",
                    "name": "solve_rows_cyclic_transposed",
                    "template": solve_cyclic_rows_kernel,
                    "kind": "kernel",
                    "bindings": {"transposed": True},
                },
            ]
        )

        self._factory.export(
            {
                "tectonic_uplift": "kernels.tectonic_uplift",
                "uplift_baselevel": "kernels.uplift_baselevel",
                "init_erode_spl": "kernels.init_erode_spl",
                "iteration_erode_spl": "kernels.iteration_erode_spl",
            }
        )

        self.gridctx.lem = self

    def destroy(self):
        """
        Release pooled parameter storage owned by this context.

        Author: B.G (03/2026)
        """
        self._factory.params.destroy()

    def __del__(self):
        """
        Best-effort pooled resource cleanup.

        Author: B.G (03/2026)
        """
        try:
            self.destroy()
        except (AttributeError, RuntimeError):
            pass
