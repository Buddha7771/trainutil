from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager


def _check_decay(decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        msg = f"decay must be in [0, 1]; got {decay}"
        raise ValueError(msg)


class EMA:
    """Exponential moving average of named tensors.

    Parameters
    ----------
    params : Mapping[str, torch.Tensor]
        Tensors to average, keyed by name.
    decay : float, optional
        Weight kept from the previous shadow at each update, in ``[0, 1]``.

    Examples
    --------
    >>> model = torch.nn.Linear(2, 1)
    >>> ema = EMA({n: p for n, p in model.named_parameters() if p.requires_grad})
    >>> ema.update(model.state_dict())
    >>> with ema.swap(model.state_dict()):
    ...     pass  # evaluate with the shadow

    """

    def __init__(
        self,
        params: Mapping[str, torch.Tensor],
        decay: float = 0.999,
    ) -> None:
        if not params:
            msg = "params is empty; nothing to average"
            raise ValueError(msg)
        _check_decay(decay)
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in params.items()
        }

    @torch.no_grad()
    def update(
        self,
        tensors: Mapping[str, torch.Tensor],
        decay: float | None = None,
    ) -> None:
        """Set each shadow to ``decay * shadow + (1 - decay) * tensor``.

        ``decay`` overrides the constructor value for this call only.
        """
        if decay is None:
            decay = self.decay
        _check_decay(decay)

        current = self._tracked(tensors)
        for name, shadow in self._shadow.items():
            shadow.lerp_(current[name], 1.0 - decay)

    def swap(self, tensors: Mapping[str, torch.Tensor]) -> AbstractContextManager:
        """Exchange tracked values in ``tensors`` with the shadow in place.

        Calling again undoes it. As a context manager, exit swaps back.

        Examples
        --------
        >>> ema.swap(model.state_dict())  # model holds the shadow
        >>> ema.swap(model.state_dict())  # model holds its own values again
        >>> with ema.swap(model.state_dict()):
        ...     pass  # model holds the shadow only inside the block

        """
        self._exchange(tensors)
        return _SwapGuard(self, tensors)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shadow keyed by name."""
        return dict(self._shadow)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load shadow; keys and shapes must match the tracked tensors."""
        missing = sorted(self._shadow.keys() - state_dict.keys())
        unexpected = sorted(state_dict.keys() - self._shadow.keys())
        if missing or unexpected:
            msg = (
                f"state_dict keys mismatch; missing {missing}, unexpected {unexpected}"
            )
            raise ValueError(msg)

        mismatched = [
            name
            for name, shadow in self._shadow.items()
            if state_dict[name].shape != shadow.shape
        ]
        if mismatched:
            msg = f"state_dict shapes mismatch for {mismatched[:5]}"
            raise ValueError(msg)

        with torch.no_grad():
            for name, shadow in self._shadow.items():
                shadow.copy_(state_dict[name])

    def to(self, device: str | torch.device) -> Self:
        """Move the shadow to ``device`` and return self."""
        self._shadow = {name: t.to(device) for name, t in self._shadow.items()}
        return self

    @torch.no_grad()
    def _exchange(self, tensors: Mapping[str, torch.Tensor]) -> None:
        current = self._tracked(tensors)
        for name, shadow in self._shadow.items():
            previous = current[name].detach().clone()
            current[name].copy_(shadow)
            shadow.copy_(previous)

    def _tracked(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self._shadow if name not in tensors]
        if missing:
            msg = (
                f"tensors has no entries named {missing[:5]}; "
                "pass a mapping from the unwrapped module"
            )
            raise KeyError(msg)
        return {name: tensors[name] for name in self._shadow}


class _SwapGuard:
    def __init__(self, ema: EMA, tensors: Mapping[str, torch.Tensor]) -> None:
        self._ema = ema
        self._tensors = tensors

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        self._ema.swap(self._tensors)
