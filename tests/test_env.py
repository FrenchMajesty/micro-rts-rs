import json

import numpy as np
from pettingzoo.test import parallel_api_test
import pytest

from micro_rts_rs import BatchedEnv, RelayEnv, verify_replay


def test_parallel_api():
    parallel_api_test(RelayEnv(fog=True), num_cycles=520)


def test_fog_observation_and_action_spaces():
    env = RelayEnv(fog=True)
    observations, _ = env.reset(seed=3)
    for agent, observation in observations.items():
        assert env.observation_space(agent).contains(observation)
        assert observation["credits"].tolist() == [3, -1]
        assert observation["source_mask"].sum() == 2
        actions = env.action_space(agent).sample(mask=observation["action_mask"])
        assert observation["action_mask"][np.arange(64), actions].all()
    p0 = observations["player_0"]["board"]
    assert tuple(p0[6, 6]) == (0,) * 9
    assert p0[5, 4, 0] == 1
    assert p0[3, 3, 1] == 1


def test_replay_round_trip_and_corrupt_hash():
    env = RelayEnv(record_replay=True)
    observations, _ = env.reset(seed=99)
    for _ in range(10):
        actions = {
            agent: env.action_space(agent).sample(mask=view["action_mask"])
            for agent, view in observations.items()
        }
        observations, *_ = env.step(actions)
    replay = json.loads(env.replay_json())
    assert verify_replay(env.replay_json()) == (10, replay["frames"][-1]["state_hash"])
    replay["frames"][6]["state_hash"] = "tampered"
    with pytest.raises(ValueError, match="tick 7"):
        verify_replay(json.dumps(replay))


def test_batch_and_parallel_trajectories_match():
    batched = BatchedEnv(2, fog=True)
    single = [RelayEnv(fog=True), RelayEnv(fog=True)]
    views = batched.reset()
    for env in single:
        env.reset()
    rng = np.random.default_rng(9)
    for _ in range(30):
        actions = np.zeros((2, 2, 64), dtype=np.int64)
        for match, seat, cell in np.argwhere(views["source_mask"]):
            actions[match, seat, cell] = rng.choice(np.flatnonzero(views["action_mask"][match, seat, cell]))
        views, rewards, done, truncated = batched.step(actions)
        for match, env in enumerate(single):
            paired = dict(zip(env.agents, actions[match]))
            step_views, step_rewards, term, trunc, _ = env.step(paired)
            for seat, agent in enumerate(paired):
                for key, field in views.items():
                    np.testing.assert_array_equal(field[match, seat], step_views[agent][key])
                assert rewards[match, seat] == step_rewards[agent]
                assert done[match] == term[agent]
                assert truncated[match] == trunc[agent]


def test_batch_rejects_finished_slots_atomically():
    env = BatchedEnv(2)
    idle = np.zeros((2, 2, 64), dtype=np.int64)
    env.reset()
    for _ in range(512):
        _, _, terminated, truncated = env.step(idle)
    assert not terminated.any()
    assert truncated.all()
    assert env.reset(0)["tick"].tolist() == [0, 0]
    with pytest.raises(ValueError, match="reset before stepping"):
        env.step(idle)
    env.reset(1)
    views, *_ = env.step(idle)
    assert views["tick"].tolist() == [[1, 1], [1, 1]]


def test_dense_grids_reject_float_and_wrapping_integer():
    env = RelayEnv()
    env.reset()
    for invalid in (np.zeros(64, dtype=np.float64), np.full(64, 256)):
        with pytest.raises(ValueError):
            env.step({"player_0": np.zeros(64, dtype=np.int64), "player_1": invalid})
    views, *_ = env.step({agent: np.zeros(64, dtype=np.int64) for agent in env.agents})
    assert views["player_0"]["tick"] == 1
