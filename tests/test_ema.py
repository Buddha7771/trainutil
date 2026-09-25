import pytest
import torch

from trainutil import EMA


def make_model() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    return model


def trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def test_update_moves_shadow_toward_tensors() -> None:
    model = make_model()
    ema = EMA(trainable(model), decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model.state_dict())

    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 1.1))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))


def test_only_given_names_are_tracked() -> None:
    model = make_model()
    ema = EMA({"weight": model.weight})
    with torch.no_grad():
        model.bias.fill_(5.0)

    ema.update(model.state_dict())
    ema.swap(model.state_dict())

    assert set(ema.state_dict()) == {"weight"}
    torch.testing.assert_close(model.bias, torch.tensor([5.0]))


def test_swap_exchanges_values_and_swaps_back_on_exit() -> None:
    model = make_model()
    ema = EMA(trainable(model), decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.update(model.state_dict())
    ema.swap(model.state_dict())
    torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))
    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 3.0))

    ema.swap(model.state_dict())
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))

    with ema.swap(model.state_dict()):
        torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))

    with pytest.raises(RuntimeError), ema.swap(model.state_dict()):
        raise RuntimeError
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))
    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 2.0))


def test_update_accepts_per_call_decay() -> None:
    model = make_model()
    ema = EMA(trainable(model), decay=0.999)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model.state_dict(), decay=0.0)

    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 2.0))


def test_state_dict_roundtrip_and_key_mismatch() -> None:
    model = make_model()
    ema = EMA(trainable(model), decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model.state_dict())

    fresh = EMA(trainable(make_model()))
    fresh.load_state_dict(ema.state_dict())

    torch.testing.assert_close(fresh.state_dict()["weight"], torch.full((1, 2), 2.0))
    with pytest.raises(ValueError, match="unexpected"):
        fresh.load_state_dict({**ema.state_dict(), "extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="shapes mismatch"):
        fresh.load_state_dict({**ema.state_dict(), "weight": torch.zeros(1)})


def test_to_moves_shadow() -> None:
    ema = EMA(trainable(make_model())).to("meta")

    assert all(t.device.type == "meta" for t in ema.state_dict().values())
