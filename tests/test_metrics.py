import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from trainutil import MetricAccumulator


def test_flush_returns_weighted_mean_and_clears() -> None:
    metrics = MetricAccumulator()
    metrics.add({"loss": 1.0, "lddt": 0.5}, weight=1)
    metrics.add({"loss": torch.tensor(4.0)}, weight=3)

    assert metrics.flush() == pytest.approx({"loss": 3.25, "lddt": 0.5})
    assert metrics.flush() == {}


def test_zero_weight_excludes_nan_values() -> None:
    metrics = MetricAccumulator()
    metrics.add({"loss": 2.0})
    metrics.add({"loss": torch.tensor(float("nan"))}, weight=0)

    assert metrics.flush() == {"loss": 2.0}


def test_add_rejects_non_scalar_tensor() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        MetricAccumulator().add({"loss": torch.ones(2)})


def _run_rank(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    metrics = MetricAccumulator()
    if rank == 0:
        metrics.add({"loss": 1.0}, weight=1)
    else:
        metrics.add({"loss": torch.tensor(4.0), "extra": 2.0}, weight=3)
    first = metrics.flush()
    empty = metrics.flush()
    Path(out_dir, f"{rank}.json").write_text(json.dumps([first, empty]))
    dist.destroy_process_group()


def test_flush_reduces_across_ranks_with_mismatched_keys(tmp_path: Path) -> None:
    world_size = 2
    mp.spawn(
        _run_rank,
        args=(world_size, str(tmp_path / "init"), str(tmp_path)),
        nprocs=world_size,
    )

    results = [json.loads((tmp_path / f"{r}.json").read_text()) for r in range(2)]
    for first, empty in results:
        assert first == pytest.approx({"loss": 3.25, "extra": 2.0})
        assert empty == {}
