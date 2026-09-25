# trainutil

Simple, composable PyTorch training utilities.

- **Batch**: declare a batch as a dataclass and collate samples into it.
- **EMA**: exponential moving average you can swap into the model for evaluation.
- **Metrics**: weighted running means reduced across DDP ranks in one call.

## Installation

```bash
pip install trainutil
```

## Usage

### Batch

```python
@dataclass
class Batch(BaseBatch):
    coords: torch.Tensor
    name: list[str]

a = Batch.from_sample(coords=torch.randn(5, 3), name="a")
b = Batch.from_sample(coords=torch.randn(3, 3), name="b")
batch = Batch.collate([a, b])

batch.coords.shape  # (2, 5, 3)
batch.name          # ['a', 'b']
batch[1]            # Batch of size one
batch.to("cuda")    # every tensor field moves
```

### EMA

```python
ema = EMA({n: p for n, p in model.named_parameters() if p.requires_grad}, decay=0.999)

optimizer.step()
ema.update(model.state_dict())

with ema.swap(model.state_dict()):  # model holds the EMA weights inside the block
    validate(model)
```

With DDP, pass the unwrapped module (`model.module`). `ema.state_dict()` uses the same
keys as the model, so `model.load_state_dict(ckpt["ema"], strict=False)` loads the EMA
weights for inference.

### Metrics

```python
metrics = MetricAccumulator()
metrics.add({"loss": loss, "lddt": lddt}, weight=batch.batch_size)

means = metrics.flush()  # weighted mean over all ranks; call on every rank
if rank == 0:
    wandb.log(means, step=step)
```
