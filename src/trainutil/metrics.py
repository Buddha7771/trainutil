from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Mapping


class MetricAccumulator:
    """Weighted running mean of named scalar metrics, reduced across DDP ranks.

    Examples
    --------
    >>> metrics = MetricAccumulator()
    >>> metrics.add({"loss": 1.0}, weight=1)
    >>> metrics.add({"loss": torch.tensor(4.0)}, weight=3)
    >>> metrics.flush()
    {'loss': 3.25}

    """

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor | float] = {}
        self._weights: dict[str, float] = {}

    def add(
        self,
        values: Mapping[str, float | torch.Tensor],
        weight: float = 1.0,
    ) -> None:
        """Add each metric weighted by ``weight``, such as the batch size."""
        if weight == 0:
            return
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    msg = f"metric {name!r} must be a scalar; got shape {value.shape}"
                    raise ValueError(msg)
                value = value.detach().reshape(()).to(torch.float64)  # noqa: PLW2901
            self._sums[name] = self._sums.get(name, 0.0) + value * weight
            self._weights[name] = self._weights.get(name, 0.0) + weight

    def flush(self) -> dict[str, float]:
        """Return the weighted mean of each metric over all ranks, then clear.

        Under DDP all ranks communicate to combine their metrics, so every rank must
        call this.
        """
        names = sorted(self._sums)
        distributed = dist.is_available() and dist.is_initialized()
        if distributed and dist.get_world_size() > 1:
            gathered: list[list[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, names)
            names = sorted({name for ranks in gathered if ranks for name in ranks})

        if not names:
            self._clear()
            return {}

        device = torch.device("cpu")
        if distributed and dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())

        sums = torch.stack(
            [
                torch.as_tensor(self._sums.get(name, 0.0), dtype=torch.float64)
                .reshape(())
                .to(device)
                for name in names
            ]
        )
        weights = torch.tensor(
            [self._weights.get(name, 0.0) for name in names],
            dtype=torch.float64,
            device=device,
        )
        totals = torch.stack([sums, weights])
        if distributed and dist.get_world_size() > 1:
            dist.all_reduce(totals)

        sums_list, weights_list = totals.cpu().tolist()
        self._clear()
        return {
            name: s / w
            for name, s, w in zip(names, sums_list, weights_list, strict=True)
            if w > 0
        }

    def _clear(self) -> None:
        self._sums.clear()
        self._weights.clear()
