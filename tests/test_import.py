"""
Tier 1: every public module imports, with no backend init.

Catches packaging / relative-import breakage (the kind a subtree move or a
renamed symbol introduces) before any GPU work is attempted.

Author: B.G (08/2026)
"""

import importlib

import pytest

import pyfastflow

_MODULES = [
    "pyfastflow.core.context",
    "pyfastflow.core.pool",
    "pyfastflow.grid",
    "pyfastflow.noise",
    "pyfastflow.flow",
    "pyfastflow.ops",
    "pyfastflow.visu",
]


@pytest.mark.parametrize("name", _MODULES)
def test_import(name):
    importlib.import_module(name)


def test_version():
    assert isinstance(pyfastflow.__version__, str) and pyfastflow.__version__


def test_lazy_submodules_resolve():
    for name in pyfastflow._LAZY_SUBMODULES:
        assert getattr(pyfastflow, name) is not None
