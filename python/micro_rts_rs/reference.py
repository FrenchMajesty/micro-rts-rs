"""Run the pinned Java GridNet client as a headless subprocess."""

from contextlib import contextmanager
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from types import SimpleNamespace

import numpy as np
import torch

from .gridnet import (
    CHANNELS, GROUPS, MASK_WIDTH, PLANES, GridNetInput,
    random_gridnet_actions, validate_gridnet_actions,
)
from .policy import JAVA_REVISION, PY_REVISION

MAP_8 = "maps/8x8/basesWorkers8x8A.xml"
MAP_16 = "maps/16x16/basesWorkers16x16A.xml"
REFERENCE_LIMIT = 2000
_HEADER = struct.Struct(">i?b")


def _require_revision(directory, expected):
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"{directory} is at {actual}; expected {expected}")


def _java_executable(name):
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / name
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"{candidate} is missing from JAVA_HOME")
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"{name} is required; set JAVA_HOME to the reference JDK")
    return executable


@contextmanager
def reference_toolchain(checkout):
    checkout = Path(checkout).resolve()
    simulation = checkout / "gym_microrts/microrts"
    _require_revision(checkout, PY_REVISION)
    _require_revision(simulation, JAVA_REVISION)
    jar = simulation / "microrts.jar"
    if not jar.is_file():
        raise FileNotFoundError(f"{jar} is missing; run upstream build.sh with JDK 8")
    java = _java_executable("java")
    compiler = _java_executable("javac")
    sys.path.insert(0, str(checkout))
    try:
        from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

        with TemporaryDirectory(prefix="relay-microrts-") as classes:
            subprocess.run([
                compiler, "-cp", str(jar), "-d", classes,
                str(Path(__file__).with_name("GridNetBridge.java")),
            ], check=True)
            command = [java, "-Djava.awt.headless=true", "-cp", os.pathsep.join((classes, str(jar))), "GridNetBridge"]
            yield simulation, command, MicroRTSGridModeVecEnv._encode_obs
    finally:
        sys.path.remove(str(checkout))


class ReferenceMatch:
    def __init__(self, process, log, board_size, encode):
        self.process = process
        self.log = log
        self.size = board_size
        self.encode = encode
        self.layout = SimpleNamespace(
            height=board_size,
            width=board_size,
            num_planes=PLANES,
            num_planes_len=len(PLANES),
            num_planes_prefix_sum=np.cumsum([0, *PLANES]),
        )
        self.inputs = None

    def _failure(self):
        self.log.seek(0)
        diagnostic = self.log.read().decode(errors="replace").strip()
        return RuntimeError(f"Java reference process stopped (exit {self.process.poll()})\n{diagnostic}".rstrip())

    def _exchange(self, payload):
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise self._failure() from error
        cells = self.size ** 2
        plane_bytes = cells * len(PLANES) * 4
        mask_bytes = cells * (1 + MASK_WIDTH)
        size = _HEADER.size + 2 * (plane_bytes + mask_bytes)
        response = self.process.stdout.read(size)
        if len(response) != size:
            raise self._failure()
        tick, terminal, winner = _HEADER.unpack_from(response)
        inputs = []
        offset = _HEADER.size
        for _ in range(2):
            raw = np.frombuffer(response, dtype=">i4", count=cells * len(PLANES), offset=offset)
            board = self.encode(self.layout, raw.reshape(len(PLANES), self.size, self.size))
            mask = np.frombuffer(response, dtype=np.uint8, count=mask_bytes, offset=offset + plane_bytes)
            mask = mask.astype(bool).reshape(cells, 1 + MASK_WIDTH)
            inputs.append(GridNetInput(board, mask[:, 0], mask[:, 1:]))
            offset += plane_bytes + mask_bytes
        self.inputs = tuple(inputs)
        return self.inputs, tick, terminal, winner

    def reset(self):
        return self._exchange(b"r")[0]

    def step(self, actions):
        if self.inputs is None or len(actions) != 2:
            raise ValueError("reset before stepping and provide both players' orders")
        parts = [b"s"]
        for rows, inputs in zip(actions, self.inputs):
            validate_gridnet_actions(rows, inputs)
            active = np.flatnonzero(inputs.source_mask)
            orders = np.column_stack((active, rows[active])).astype(">i4")
            parts.extend((struct.pack(">i", len(active)), orders.tobytes()))
        return self._exchange(b"".join(parts))


@contextmanager
def open_match(toolchain, map_path, board_size):
    simulation, command, encode = toolchain
    with TemporaryFile() as log, subprocess.Popen(
        [*command, str(simulation / map_path)],
        cwd=simulation, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
    ) as process:
        try:
            yield ReferenceMatch(process, log, board_size, encode)
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@torch.inference_mode()
def check_actor_parity(checkout, toolchain, policy, checkpoint):
    experiment = runpy.run_path(str(Path(checkout) / "experiments/ppo_gridnet.py"))
    layout = SimpleNamespace(observation_space=SimpleNamespace(shape=(16, 16, CHANNELS)))
    upstream = experiment["Agent"](layout).eval()
    upstream.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    categorical = experiment["CategoricalMasked"]
    rng = np.random.default_rng(0)
    with open_match(toolchain, MAP_16, 16) as env:
        inputs = env.reset()
        for _ in range(8):
            boards = torch.as_tensor(np.stack([seat.board for seat in inputs]), dtype=torch.float32)
            masks = torch.as_tensor(np.stack([seat.factor_mask for seat in inputs]), dtype=torch.bool)
            expected = upstream.actor(upstream.encoder(boards)).reshape(2, -1, MASK_WIDTH)
            actual = policy(boards).reshape_as(expected)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
            probabilities = policy.masked_probabilities(actual, masks)
            for result, group, mask in zip(probabilities, expected.split(GROUPS, -1), masks.split(GROUPS, -1)):
                distribution = categorical(logits=group, masks=mask, mask_value=upstream.mask_value)
                torch.testing.assert_close(result, distribution.probs, atol=1e-7, rtol=1e-6)
            actions = [random_gridnet_actions(seat, rng) for seat in inputs]
            inputs, _, _, _ = env.step(actions)
