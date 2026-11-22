import numpy as np
import pytest

from micro_rts_rs import RelayEnv
from micro_rts_rs.gridnet import decode_native, project_native, random_gridnet_actions


def test_full_observation_and_legacy_masks():
    views, _ = RelayEnv().reset()
    native = views["player_0"]
    gridnet = project_native(native)
    assert gridnet.board.shape == (8, 8, 29)
    assert (gridnet.board.sum(-1) == 6).all()
    assert gridnet.factor_mask.shape == (64, 78)
    assert sorted(np.flatnonzero(gridnet.source_mask)) == [9, 10]
    assert not gridnet.factor_mask[0].any()
    assert gridnet.board[1, 1, 4] == 1   # Clipped base HP.
    assert gridnet.board[1, 2, 17] == 1  # Worker type.
    assert gridnet.board[2, 3, 28] == 1  # Wall.
    assert gridnet.factor_mask[9, 25]    # Worker production.
    assert not gridnet.factor_mask[9, 26]


def test_native_projection_decodes_only_available_actions():
    env = RelayEnv()
    views, _ = env.reset()
    rng = np.random.default_rng(8)
    for _ in range(30):
        actions = {}
        for agent, native in views.items():
            inputs = project_native(native)
            selected = random_gridnet_actions(inputs, rng)
            decoded = decode_native(selected, native, inputs)
            assert native["action_mask"][np.arange(64), decoded].all()
            actions[agent] = decoded
        views, *_ = env.step(actions)
    native = views["player_0"]
    inputs = project_native(native)
    selected = random_gridnet_actions(inputs, rng)
    cell = np.flatnonzero(inputs.source_mask)[0]
    selected[cell, 0] = 99
    with pytest.raises(ValueError, match="invalid GridNet action type"):
        decode_native(selected, native, inputs)


def test_fog_requires_a_different_policy_profile():
    views, _ = RelayEnv(fog=True).reset()
    with pytest.raises(ValueError, match="requires full observation"):
        project_native(views["player_0"])
