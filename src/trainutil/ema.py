from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager

    from torch import nn


def _check_decay(decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        msg = f"decay must be in [0, 1]; got {decay}"
        raise ValueError(msg)


class EMA:
    """Exponential moving average of a module's trainable parameters.

    Parameters
    ----------
    module : nn.Module
        Module whose ``requires_grad`` parameters are averaged. Pass the unwrapped
        module, not a DDP or ``torch.compile`` wrapper, so that names match the
        module's ``state_dict``.
    decay : float, optional
        Weight kept from the previous shadow at each update, in ``[0, 1]``.

    Examples
    --------
    >>> model = torch.nn.Linear(2, 1)
    >>> ema = EMA(model)
    >>> ema.update()
    >>> with ema.swap():
    ...     pass  # evaluate with the shadow

    """

    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self._params = {
            name: param
            for name, param in module.named_parameters()
            if param.requires_grad
        }
        if not self._params:
            msg = "module has no trainable parameters; nothing to average"
            raise ValueError(msg)
        _check_decay(decay)
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone() for name, param in self._params.items()
        }

    @torch.no_grad()
    def update(self, decay: float | None = None) -> None:
        """Set each shadow to ``decay * shadow + (1 - decay) * param``.

        ``decay`` overrides the constructor value for this call only.
        """
        if decay is None:
            decay = self.decay
        _check_decay(decay)

        for name, shadow in self._shadow.items():
            shadow.lerp_(self._params[name], 1.0 - decay)

    def swap(self) -> AbstractContextManager:
        """Exchange the parameters with the shadow in place.

        Calling again undoes it. As a context manager, exit swaps back.

        Examples
        --------
        >>> ema.swap()  # model holds the shadow
        >>> ema.swap()  # model holds its own values again
        >>> with ema.swap():
        ...     pass  # model holds the shadow only inside the block

        """
        self._exchange()
        return _SwapGuard(self)

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
    def _exchange(self) -> None:
        for name, shadow in self._shadow.items():
            param = self._params[name]
            previous = param.detach().clone()
            param.copy_(shadow)
            shadow.copy_(previous)


class _SwapGuard:
    def __init__(self, ema: EMA) -> None:
        self._ema = ema

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        self._ema.swap()
