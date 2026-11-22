"""Observation and action adapters for the pinned, fully observed GridNet profile."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from ._native import ACTIONS, CELLS, WIDTH

GROUPS = (6, 4, 4, 4, 4, 7, 49)
OFFSETS = tuple(sum(GROUPS[:index]) for index in range(len(GROUPS)))
PLANES = (5, 5, 3, 8, 6, 2)
CHANNELS = sum(PLANES)
MASK_WIDTH = sum(GROUPS)
_TYPE_GROUPS = ((), (1,), (2,), (3,), (4, 5), (6,))
_ATTACK_CELLS = (17, 25, 31, 23)
# GridNet unit types in this profile are Resource, Base, Barracks, Worker,
# Light, Heavy, Ranged. Observations reserve index zero for the empty cell.
_KIND_FEATURES = np.array([0, 1, 2, 4, 5])
_ORDER_FEATURES = np.array([0, 1, 2, 3, 4, 4, 5])
_ONE_HOT = tuple(np.eye(size, dtype=np.int32) for size in PLANES)


def _gridnet_commands():
    commands = [((0, 0),)]
    for action_type in (1, 2, 3):
        commands.extend(((0, action_type), (action_type, direction)) for direction in range(4))
    for unit_type in (3, 4):  # Worker and Light, without the empty-cell offset.
        commands.extend(((0, 4), (4, direction), (5, unit_type)) for direction in range(4))
    commands.extend(((0, 5), (6, cell)) for cell in _ATTACK_CELLS)
    return tuple(commands)


_NATIVE_COMMANDS = _gridnet_commands()
_NATIVE_LOOKUP = {
    tuple(value for _, value in command): index for index, command in enumerate(_NATIVE_COMMANDS)
}


@dataclass(frozen=True)
class GridNetInput:
    board: np.ndarray
    source_mask: np.ndarray
    factor_mask: np.ndarray


def project_native(observation):
    """Project a fully observed Relay view, leaving relay and player banks out."""
    board = observation["board"]
    if not board[..., 2].all():
        raise ValueError("the published GridNet checkpoint requires full observation")
    raw_features = (
        board[..., 5],
        board[..., 6],
        board[..., 3],
        _KIND_FEATURES[board[..., 4]],
        _ORDER_FEATURES[board[..., 7]],
        board[..., 0],
    )
    encoded = np.concatenate([
        one_hot[np.clip(values, 0, len(one_hot) - 1)]
        for values, one_hot in zip(raw_features, _ONE_HOT)
    ], axis=-1)

    source = observation["source_mask"].astype(bool)
    native_mask = observation["action_mask"].astype(bool) & source[:, None]
    factors = np.zeros((CELLS, MASK_WIDTH), dtype=bool)
    for action, command in enumerate(_NATIVE_COMMANDS):
        for group, value in command:
            factors[:, OFFSETS[group] + value] |= native_mask[:, action]
    return GridNetInput(encoded, source, factors)


def _command_key(row):
    action_type = row[0]
    return (action_type, *(row[group] for group in _TYPE_GROUPS[action_type]))


def validate_gridnet_actions(actions, inputs):
    """Check active sources and the factors used by each chosen action type."""
    actions = np.asarray(actions)
    if actions.shape != (len(inputs.source_mask), len(GROUPS)) or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("expected one seven-factor integer action per cell")
    for cell in np.flatnonzero(inputs.source_mask):
        row = actions[cell]
        if row[0] < 0 or row[0] >= GROUPS[0]:
            raise ValueError(f"invalid GridNet action type at cell {cell}")
        for factor in (0, *_TYPE_GROUPS[row[0]]):
            value = row[factor]
            if value < 0 or value >= GROUPS[factor] or not inputs.factor_mask[cell, OFFSETS[factor] + value]:
                raise ValueError(f"masked GridNet choice at cell {cell}, factor {factor}")


def decode_native(actions, observation, inputs):
    """Decode sampled GridNet factors and validate the composed native actions."""
    if inputs.board.shape != (WIDTH, WIDTH, CHANNELS):
        raise ValueError("expected an 8×8 native projection")
    validate_gridnet_actions(actions, inputs)
    decoded = np.zeros(CELLS, dtype=np.int64)
    for cell in np.flatnonzero(inputs.source_mask):
        action = _NATIVE_LOOKUP.get(_command_key(actions[cell]))
        if action is None or not observation["action_mask"][cell, action]:
            raise ValueError(f"unsupported native action at cell {cell}")
        decoded[cell] = action
    return decoded


def random_native_actions(observation, rng):
    result = np.zeros(CELLS, dtype=np.int64)
    for cell in np.flatnonzero(observation["source_mask"]):
        result[cell] = rng.choice(np.flatnonzero(observation["action_mask"][cell]))
    return result


def random_gridnet_actions(inputs, rng):
    """Sample uniformly over complete masked commands independently per unit."""
    result = np.zeros((len(inputs.source_mask), len(GROUPS)), dtype=np.int64)
    for cell in np.flatnonzero(inputs.source_mask):
        mask = inputs.factor_mask[cell]
        commands = []
        for action_type in np.flatnonzero(mask[:GROUPS[0]]):
            relevant = _TYPE_GROUPS[action_type]
            allowed = [np.flatnonzero(mask[OFFSETS[i]:OFFSETS[i] + GROUPS[i]]) for i in relevant]
            commands.extend((action_type, relevant, values) for values in product(*allowed))
        if not commands:
            raise ValueError(f"GridNet source {cell} has no legal command")
        action_type, relevant, values = commands[rng.integers(len(commands))]
        result[cell, 0] = action_type
        result[cell, list(relevant)] = values
    return result
