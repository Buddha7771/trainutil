import pytest
import torch

from trainutil import EMA


def make_model() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    return model


def test_update_moves_shadow_toward_params() -> None:
    model = make_model()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update()

    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 1.1))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))


def test_only_trainable_params_are_tracked() -> None:
    model = make_model()
    model.bias.requires_grad = False
    ema = EMA(model)
    with torch.no_grad():
        model.bias.fill_(5.0)

    ema.update()
    ema.swap()

    assert set(ema.state_dict()) == {"weight"}
    torch.testing.assert_close(model.bias, torch.tensor([5.0]))


def test_module_without_trainable_params_raises() -> None:
    model = make_model().requires_grad_(requires_grad=False)

    with pytest.raises(ValueError, match="no trainable parameters"):
        EMA(model)


def test_swap_exchanges_values_and_swaps_back_on_exit() -> None:
    model = make_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.update()
    ema.swap()
    torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))
    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 3.0))

    ema.swap()
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))

    with ema.swap():
        torch.testing.assert_close(model.weight, torch.full((1, 2), 2.0))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))

    with pytest.raises(RuntimeError), ema.swap():
        raise RuntimeError
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))
    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 2.0))


def test_update_accepts_per_call_decay() -> None:
    model = make_model()
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(decay=0.0)

    torch.testing.assert_close(ema.state_dict()["weight"], torch.full((1, 2), 2.0))


def test_state_dict_roundtrip_and_key_mismatch() -> None:
    model = make_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update()

    fresh = EMA(make_model())
    fresh.load_state_dict(ema.state_dict())

    torch.testing.assert_close(fresh.state_dict()["weight"], torch.full((1, 2), 2.0))
    with pytest.raises(ValueError, match="unexpected"):
        fresh.load_state_dict({**ema.state_dict(), "extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="shapes mismatch"):
        fresh.load_state_dict({**ema.state_dict(), "weight": torch.zeros(1)})


def test_state_dict_keys_match_module() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))

    assert set(EMA(model).state_dict()) == set(model.state_dict())


def test_to_moves_shadow() -> None:
    ema = EMA(make_model()).to("meta")

    assert all(t.device.type == "meta" for t in ema.state_dict().values())
