import pytest

from aorta_surrogate.data.convert_stanford import select_evenly_spaced_timesteps


def test_selects_cycle_endpoints_and_registered_phases():
    available = list(range(22480, 25681, 5))
    selected = select_evenly_spaced_timesteps(available, 21)

    assert len(selected) == 21
    assert selected[0] == 22480
    assert selected[-1] == 25680
    assert all(value in available for value in selected)


def test_rejects_too_few_available_phases():
    with pytest.raises(ValueError, match="only 3 available"):
        select_evenly_spaced_timesteps([1, 2, 3], 4)
