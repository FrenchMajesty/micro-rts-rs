"""PettingZoo and batched interfaces for the native simulator."""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ._native import ACTIONS, BOARD_FEATURES, CELLS, MAX_TICKS, WIDTH, CoreBatch, CoreMatch


class _GridActions(spaces.MultiDiscrete):
    def sample(self, mask=None, probability=None):
        if isinstance(mask, np.ndarray):
            if mask.shape != (CELLS, ACTIONS):
                raise ValueError(f"expected an action mask of shape {(CELLS, ACTIONS)}")
            mask = tuple(mask.astype(np.int8, copy=False))
        if probability is None:
            return super().sample(mask=mask)
        return super().sample(mask=mask, probability=probability)


def _observation(raw):
    board, credits, tick, mask, sources = raw
    return {
        "board": np.asarray(board, dtype=np.int32).reshape(WIDTH, WIDTH, BOARD_FEATURES),
        "credits": np.asarray(credits, dtype=np.int32),
        "tick": np.asarray(tick, dtype=np.int32),
        "action_mask": np.frombuffer(mask, dtype=np.int8).reshape(CELLS, ACTIONS),
        "source_mask": np.frombuffer(sources, dtype=np.int8),
    }


def _choices(value, shape):
    value = np.asarray(value)
    if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"expected integer action grids of shape {shape}")
    if np.any(value < 0) or np.any(value >= ACTIONS):
        raise ValueError(f"expected all action choices in [0, {ACTIONS})")
    return value.tolist()


class RelayEnv(ParallelEnv):
    """Two players commit their row-major action grids on the same tick."""

    metadata = {"name": "relay_8x8_v1", "render_modes": [], "is_parallelizable": True}

    def __init__(self, *, fog=False, record_replay=False):
        self.possible_agents = ["player_0", "player_1"]
        self.agents = []
        self.render_mode = None
        self._match = CoreMatch(fog, record_replay)
        self._action_spaces = {
            agent: _GridActions(np.full(CELLS, ACTIONS)) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict({
                "board": spaces.Box(0, 12, shape=(WIDTH, WIDTH, BOARD_FEATURES), dtype=np.int32),
                "credits": spaces.Box(-1, np.iinfo(np.int32).max, shape=(2,), dtype=np.int32),
                "tick": spaces.Box(0, MAX_TICKS, shape=(), dtype=np.int32),
                "action_mask": spaces.MultiBinary((CELLS, ACTIONS)),
                "source_mask": spaces.MultiBinary(CELLS),
            })
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        if seed is not None:
            for index, agent in enumerate(self.agents):
                self.action_space(agent).seed(seed + index)
        observations = dict(zip(self.agents, map(_observation, self._match.reset())))
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        if set(actions) != set(self.agents):
            raise ValueError("submit one action grid for each active player")
        if not self.agents:
            return {}, {}, {}, {}, {}
        values = _choices([actions[agent] for agent in self.agents], (2, CELLS))
        raw, rewards, terminated, truncated = self._match.step(values)
        players = self.agents
        if terminated or truncated:
            self.agents = []
        return (
            dict(zip(players, map(_observation, raw))),
            dict(zip(players, rewards)),
            {agent: terminated for agent in players},
            {agent: truncated for agent in players},
            {agent: {} for agent in players},
        )

    def replay_json(self):
        """Export the recorded joint orders and full-state hashes for trusted use."""
        return self._match.replay_json()


parallel_env = RelayEnv


def _stack_views(matches):
    views = [[_observation(raw) for raw in match] for match in matches]
    return {
        field: np.stack([np.stack([view[field] for view in match]) for match in views])
        for field in views[0][0]
    }


class BatchedEnv:
    """Rust-backed collection of matches with explicit per-slot resets.

    Step takes integer actions shaped (matches, 2, 64). Observations have
    leading (matches, 2) dimensions. An indexed reset returns a single slot
    with leading (2,). Finished slots must be reset before the next joint step.
    """

    def __init__(self, size, *, fog=False):
        self._core = CoreBatch(size, fog)
        self.size = size

    def reset(self, index=None):
        views = _stack_views(self._core.reset(index))
        if index is not None:
            return {field: value[0] for field, value in views.items()}
        return views

    def step(self, actions):
        values = _choices(actions, (self.size, 2, CELLS))
        raw, rewards, terminations, truncations = zip(*self._core.step(values))
        return (
            _stack_views(raw),
            np.asarray(rewards, dtype=np.int8),
            np.asarray(terminations, dtype=np.bool_),
            np.asarray(truncations, dtype=np.bool_),
        )
