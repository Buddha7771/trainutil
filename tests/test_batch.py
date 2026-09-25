from dataclasses import dataclass

import numpy as np
import pytest
import torch

from trainutil import BaseBatch
from trainutil.batch import move_to_device, pad_and_collate


@dataclass
class Metadata(BaseBatch):
    confidence: torch.Tensor


@dataclass
class Batch(BaseBatch):
    features: torch.Tensor
    names: list[str]
    metadata: Metadata
    mask: torch.Tensor | None = None


def test_pad_and_collate_pads_to_largest_shape() -> None:
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second = torch.tensor([[5.0, 6.0]])

    result = pad_and_collate([first, second])

    assert result.shape == (2, 2, 2)
    torch.testing.assert_close(result[0], first)
    torch.testing.assert_close(result[1], torch.tensor([[5.0, 6.0], [0.0, 0.0]]))


def test_pad_and_collate_rejects_mixed_ranks() -> None:
    with pytest.raises(ValueError, match="same number of dimensions"):
        pad_and_collate([torch.ones(2), torch.ones(2, 2)])


def test_from_sample_and_batch_size() -> None:
    batch = Batch.from_sample(
        features=torch.tensor([1.0, 2.0]),
        names="first",
        metadata=Metadata.from_sample(confidence=torch.tensor(0.8)),
        mask=None,
    )

    assert batch.features.shape == (1, 2)
    assert batch.names == ["first"]
    assert batch.metadata.confidence.shape == (1,)
    assert batch.batch_size == 1


def test_collate_nested_batches_and_preserve_padding() -> None:
    first = Batch.from_sample(
        features=torch.tensor([[1.0], [2.0]]),
        names="first",
        metadata=Metadata.from_sample(confidence=torch.tensor(0.8)),
    )
    second = Batch.from_sample(
        features=torch.tensor([[3.0]]),
        names="second",
        metadata=Metadata.from_sample(confidence=torch.tensor(0.6)),
    )

    batch = Batch.collate([first, second])

    assert batch.features.shape == (2, 2, 1)
    assert batch.names == ["first", "second"]
    torch.testing.assert_close(batch.features[1, 1], torch.tensor([0.0]))
    torch.testing.assert_close(
        batch.metadata.confidence,
        torch.tensor([0.8, 0.6]),
    )


def test_collate_rejects_mixed_field_types() -> None:
    first = Batch.from_sample(
        features=torch.tensor([1.0]),
        names="first",
        metadata=Metadata.from_sample(confidence=torch.tensor(0.8)),
        mask=torch.ones(1, dtype=torch.bool),
    )
    second = Batch.from_sample(
        features=torch.tensor([2.0]),
        names="second",
        metadata=Metadata.from_sample(confidence=torch.tensor(0.6)),
        mask=None,
    )

    with pytest.raises(TypeError, match="inconsistent types"):
        Batch.collate([first, second])


def test_integer_index_preserves_batch_dimension() -> None:
    batch = Batch(
        features=torch.arange(6).reshape(3, 2),
        names=["a", "b", "c"],
        metadata=Metadata(confidence=torch.tensor([0.1, 0.2, 0.3])),
    )

    item = batch[1]

    assert item.features.shape == (1, 2)
    assert item.names == ["b"]
    assert item.metadata.confidence.shape == (1,)
    assert item.batch_size == 1


def test_slice_and_negative_index_select_matching_elements() -> None:
    batch = Batch(
        features=torch.arange(6).reshape(3, 2),
        names=["a", "b", "c"],
        metadata=Metadata(confidence=torch.tensor([0.1, 0.2, 0.3])),
    )

    tail = batch[1:]
    last = batch[-1]

    assert tail.names == ["b", "c"]
    torch.testing.assert_close(tail.features, batch.features[1:])
    torch.testing.assert_close(tail.metadata.confidence, torch.tensor([0.2, 0.3]))
    assert last.names == ["c"]
    torch.testing.assert_close(last.features, batch.features[2:3])


def test_to_moves_nested_tensors_and_keeps_other_fields() -> None:
    batch = Batch(
        features=torch.ones(2, 3),
        names=["a", "b"],
        metadata=Metadata(confidence=torch.ones(2)),
    )

    moved = batch.to("meta")

    assert moved.features.device.type == "meta"
    assert moved.metadata.confidence.device.type == "meta"
    assert moved.names == ["a", "b"]
    assert moved.mask is None


def test_move_to_device_recurses_into_dict_and_list() -> None:
    data = {"tensor": torch.ones(2), "nested": [torch.zeros(1), "unchanged"]}

    moved = move_to_device(data, "meta")

    tensor = moved["tensor"]
    nested = moved["nested"]
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(nested, list)
    assert isinstance(nested[0], torch.Tensor)
    assert tensor.device.type == "meta"
    assert nested[0].device.type == "meta"
    assert nested[1] == "unchanged"


def test_inconsistent_field_sizes_raise_value_error_on_construction() -> None:
    with pytest.raises(ValueError, match="not consistent"):
        Batch(
            features=torch.ones(2, 3),
            names=["only-one"],
            metadata=Metadata(confidence=torch.ones(2)),
        )


def test_integer_like_index_keeps_batch_dimension() -> None:
    batch = Batch(
        features=torch.arange(6).reshape(3, 2),
        names=["a", "b", "c"],
        metadata=Metadata(confidence=torch.tensor([0.1, 0.2, 0.3])),
    )

    item = batch[np.int64(1)]

    assert item.names == ["b"]
    assert item.features.shape == (1, 2)


def test_zero_dim_tensor_field_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no batch dimension"):
        Batch(
            features=torch.tensor(1.0),
            names=["a"],
            metadata=Metadata(confidence=torch.ones(1)),
        )
