from __future__ import annotations

import operator
from dataclasses import dataclass, fields
from typing import Any, SupportsIndex, TypeVar, overload

import torch
from torch.nn import functional
from typing_extensions import Self

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
BatchT = TypeVar("BatchT", bound="BaseBatch")


def pad_and_collate(
    tensors: list[torch.Tensor],
    batch_dim: int | None = None,
) -> torch.Tensor:
    """Pad tensors to a common shape, then stack or concatenate them.

    Every dimension is zero-padded up to the largest size found across
    ``tensors``. The dimension given by ``batch_dim`` is left as is and used for
    concatenation instead.

    Parameters
    ----------
    tensors : list[torch.Tensor]
        Tensors with the same number of dimensions.
    batch_dim : int, optional
        Dimension to concatenate along. If None, tensors are stacked into a new
        leading dimension.

    Returns
    -------
    torch.Tensor
        Padded and collated tensor.

    Raises
    ------
    ValueError
        If ``tensors`` is empty or the tensors differ in rank.
    IndexError
        If ``batch_dim`` is out of range for the tensor rank.

    """
    if not tensors:
        msg = "tensors cannot be empty"
        raise ValueError(msg)

    rank = tensors[0].ndim
    if any(tensor.ndim != rank for tensor in tensors):
        shapes = [tuple(tensor.shape) for tensor in tensors]
        msg = f"All tensors must have the same number of dimensions; got {shapes}"
        raise ValueError(msg)

    if batch_dim is not None and not -rank <= batch_dim < rank:
        msg = f"batch_dim {batch_dim} is out of range for tensors with rank {rank}"
        raise IndexError(msg)

    normalized_batch_dim = batch_dim % rank if batch_dim is not None else None
    max_shape = [max(tensor.shape[dim] for tensor in tensors) for dim in range(rank)]
    padded = []
    for tensor in tensors:
        padding = []
        for dim in reversed(range(rank)):
            missing = (
                0 if dim == normalized_batch_dim else max_shape[dim] - tensor.shape[dim]
            )
            padding.extend((0, missing))
        padded.append(functional.pad(tensor, padding) if any(padding) else tensor)

    if batch_dim is None:
        return torch.stack(padded)
    return torch.cat(padded, dim=batch_dim)


@overload
def move_to_device(data: torch.Tensor, device: str | torch.device) -> torch.Tensor: ...


@overload
def move_to_device(data: dict[K, V], device: str | torch.device) -> dict[K, V]: ...


@overload
def move_to_device(data: list[T], device: str | torch.device) -> list[T]: ...


@overload
def move_to_device(
    data: tuple[T, ...],
    device: str | torch.device,
) -> tuple[T, ...]: ...


@overload
def move_to_device(data: BatchT, device: str | torch.device) -> BatchT: ...


def move_to_device(data: Any, device: str | torch.device) -> Any:
    """Move tensors nested in ``data`` to a device.

    Parameters
    ----------
    data : Any
        A tensor, a :class:`BaseBatch`, or a dict, list, or tuple containing
        them at any depth. Other objects are returned unchanged.
    device : str or torch.device
        Target device.

    Returns
    -------
    Any
        ``data`` with every tensor moved to ``device``. Containers are rebuilt
        as plain dict, list, or tuple.

    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, BaseBatch):
        return data.to(device)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(value, device) for value in data)
    return data


@dataclass
class BaseBatch:
    """Base class for dataclass batches with nested tensor fields.

    Subclasses declare their fields with ``@dataclass``. Supported field types
    are ``torch.Tensor``, another :class:`BaseBatch`, ``list``, and ``None``.
    Tensors and nested batches share a leading batch dimension; lists hold one
    item per batch element.

    Examples
    --------
    >>> @dataclass
    ... class MyBatch(BaseBatch):
    ...     coords: torch.Tensor
    ...     names: list[str]
    >>> sample = MyBatch.from_sample(coords=torch.zeros(3, 3), names="a")
    >>> MyBatch.collate([sample, sample]).batch_size
    2

    """

    def __post_init__(self) -> None:
        """Check that all fields agree on the batch size."""
        _ = self.batch_size

    def to(self, device: str | torch.device) -> Self:
        """Return a copy with every nested tensor moved to ``device``."""
        return self.__class__(
            **{
                field.name: move_to_device(getattr(self, field.name), device)
                for field in fields(self)
            },
        )

    @classmethod
    def collate(cls, batches: list[Self]) -> Self:
        """Collate batches of this exact type into one batch.

        Tensor fields are padded with :func:`pad_and_collate` along every
        dimension except the leading batch dimension, then concatenated. Nested
        batch fields are collated recursively, list fields are concatenated, and
        None fields stay None.

        Parameters
        ----------
        batches : list[Self]
            Non-empty list of instances of ``cls``.

        Returns
        -------
        Self
            Collated batch.

        Raises
        ------
        ValueError
            If ``batches`` is empty.
        TypeError
            If an item is not exactly ``cls``, a field holds mixed types across
            batches, or a field type is unsupported.

        """
        if not batches:
            msg = "batches cannot be empty"
            raise ValueError(msg)
        if any(type(batch) is not cls for batch in batches):
            types = [type(batch).__name__ for batch in batches]
            msg = f"All batches must be {cls.__name__}; got {types}"
            raise TypeError(msg)

        collated: dict[str, object] = {}
        for field in fields(cls):
            values = [getattr(batch, field.name) for batch in batches]
            first = values[0]
            if any(type(value) is not type(first) for value in values):
                types = [type(value).__name__ for value in values]
                msg = f"Field {field.name!r} has inconsistent types: {types}"
                raise TypeError(msg)

            if isinstance(first, torch.Tensor):
                collated[field.name] = pad_and_collate(values, batch_dim=0)
            elif isinstance(first, BaseBatch):
                collated[field.name] = type(first).collate(values)
            elif first is None:
                collated[field.name] = None
            elif isinstance(first, list):
                collated[field.name] = [item for value in values for item in value]
            else:
                type_name = type(first).__name__
                msg = f"Field {field.name!r} has unsupported type {type_name}"
                raise TypeError(msg)

        return cls(**collated)

    @classmethod
    def from_sample(cls, **values: object) -> Self:
        """Create a batch of size one from unbatched field values.

        Parameters
        ----------
        **values : object
            One value per field. Tensors get a new leading dimension, nested
            batches and None are kept as is, and any other value is wrapped in
            a one-item list.

        Returns
        -------
        Self
            Batch with ``batch_size == 1``.

        """
        batched: dict[str, object] = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                batched[name] = value.unsqueeze(0)
            elif isinstance(value, BaseBatch) or value is None:
                batched[name] = value
            else:
                batched[name] = [value]
        return cls(**batched)

    @property
    def batch_size(self) -> int:
        """Return the leading size shared by all fields."""
        sizes: dict[str, int] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    msg = f"Field {field.name!r} has no batch dimension (0-dim tensor)"
                    raise ValueError(msg)
                sizes[field.name] = value.shape[0]
            elif isinstance(value, BaseBatch):
                sizes[field.name] = value.batch_size
            elif isinstance(value, list):
                sizes[field.name] = len(value)
            elif value is not None:
                type_name = type(value).__name__
                msg = f"Field {field.name!r} has unsupported type {type_name}"
                raise TypeError(msg)

        if not sizes:
            return 0
        size = next(iter(sizes.values()))
        if any(field_size != size for field_size in sizes.values()):
            msg = f"Batch size is not consistent across fields: {sizes}"
            raise ValueError(msg)
        return size

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, index: SupportsIndex | slice) -> Self:
        """Select batch elements, keeping the batch dimension.

        Parameters
        ----------
        index : SupportsIndex or slice
            Integer position or range along the batch dimension. Negative
            positions count from the end.

        Returns
        -------
        Self
            Batch holding the selected elements. An integer index yields a
            batch of size one.

        Raises
        ------
        IndexError
            If an integer position is out of bounds.
        TypeError
            If ``index`` is neither a slice nor integer-like, or a field type is
            unsupported.

        """
        if not isinstance(index, slice):
            size = self.batch_size
            position = operator.index(index)
            normalized = position + size if position < 0 else position
            if not 0 <= normalized < size:
                msg = f"Index {position} is out of bounds for batch of size {size}"
                raise IndexError(msg)
            index = slice(normalized, normalized + 1)

        selected: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (torch.Tensor, BaseBatch, list)):
                selected[field.name] = value[index]
            elif value is None:
                selected[field.name] = None
            else:
                type_name = type(value).__name__
                msg = f"Field {field.name!r} has unsupported type {type_name}"
                raise TypeError(msg)
        return self.__class__(**selected)
