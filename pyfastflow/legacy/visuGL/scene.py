from __future__ import annotations

from typing import List


class Layer:
    """Dumb render layer interface.

    Implementors should provide:
    - setup(ctx, hub): compile/allocate once
    - draw(ctx, camera): per-frame render
    """

    def setup(self, ctx, hub):  # pragma: no cover - interface
        raise NotImplementedError

    def draw(self, ctx, camera):  # pragma: no cover - interface
        raise NotImplementedError


class Scene:
    def __init__(self) -> None:
        self._layers: List[Layer] = []

    def add(self, layer: Layer) -> None:
        self._layers.append(layer)

    # Internal: called by app once GL is alive
    def _setup(self, ctx, hub) -> None:
        for layer in self._layers:
            layer.setup(ctx, hub)

    def draw_all(self, ctx, camera) -> None:
        for layer in self._layers:
            layer.draw(ctx, camera)

