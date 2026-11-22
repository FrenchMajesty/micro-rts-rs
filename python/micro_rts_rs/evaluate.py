"""Measure transfer of the published GridNet checkpoint on both benchmarks."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from ._native import MAX_TICKS
from .env import RelayEnv
from .gridnet import decode_native, project_native, random_gridnet_actions, random_native_actions
from .policy import CHECKPOINT_PATH, CHECKPOINT_SHA256, GridNetPolicy


def _rngs(seed):
    return torch.Generator().manual_seed(seed), np.random.default_rng(seed)


def _native_episode(policy, seed, policy_seat):
    env = RelayEnv()
    policy_rng, opponent_rng = _rngs(seed)
    observations, _ = env.reset()
    policy_agent = env.possible_agents[policy_seat]
    opponent_agent = env.possible_agents[1 - policy_seat]
    for tick in range(1, MAX_TICKS + 1):
        view = observations[policy_agent]
        inputs = project_native(view)
        actions = {
            policy_agent: decode_native(policy.choose(inputs, policy_rng), view, inputs),
            opponent_agent: random_native_actions(observations[opponent_agent], opponent_rng),
        }
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if terminated[policy_agent] or truncated[policy_agent]:
            return rewards[policy_agent], tick, truncated[policy_agent]
    raise RuntimeError("native match exceeded its tick limit")


def _reference_episode(env, limit, policy, seed, policy_seat):
    policy_rng, opponent_rng = _rngs(seed)
    inputs = env.reset()
    for _ in range(limit):
        actions = [None, None]
        actions[policy_seat] = policy.choose(inputs[policy_seat], policy_rng)
        actions[1 - policy_seat] = random_gridnet_actions(inputs[1 - policy_seat], opponent_rng)
        inputs, tick, terminal, winner = env.step(actions)
        if terminal:
            return (0 if winner == -1 else 1 if winner == policy_seat else -1), tick, False
    return 0, tick, True


def _measure(episode, seeds, tick_limit):
    results = []
    start = perf_counter()
    for seed in seeds:
        for seat in (0, 1):
            reward, ticks, truncated = episode(seed, seat)
            results.append({
                "seed": seed,
                "policy_seat": seat,
                "outcome": {-1: "loss", 0: "draw", 1: "win"}[reward],
                "ticks": ticks,
                "truncated": truncated,
            })
    elapsed = perf_counter() - start
    counts = Counter(match["outcome"] for match in results)
    ticks = [match["ticks"] for match in results]
    return {
        "tick_limit": tick_limit,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "truncated": sum(match["truncated"] for match in results),
        "mean_ticks": mean(ticks),
        "joint_ticks_per_second": round(sum(ticks) / elapsed, 1),
        "matches": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native", "java", "both"), default="native")
    parser.add_argument("--checkpoint", type=Path, help="path to the published agent_sota.pt")
    parser.add_argument("--reference", type=Path, help="pinned MicroRTS-Py checkout with a built Java submodule")
    parser.add_argument("--seed", type=int, action="append", help="repeat to evaluate multiple seeds on both seats")
    parser.add_argument("--verify-upstream", action="store_true", help="compare 16×16 logits and probabilities first")
    args = parser.parse_args()
    needs_reference = args.engine in ("java", "both") or args.verify_upstream
    if needs_reference and args.reference is None:
        parser.error("--reference is required for Java evaluation and actor parity")
    if args.checkpoint is None and args.reference is None:
        parser.error("supply --checkpoint or --reference")
    seeds = args.seed if args.seed is not None else [7, 11]
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        parser.error("seeds must be nonnegative 63-bit integers")

    checkpoint = args.checkpoint or args.reference / CHECKPOINT_PATH
    torch.set_num_threads(1)
    policy = GridNetPolicy.from_checkpoint(checkpoint)
    report = {"checkpoint_sha256": CHECKPOINT_SHA256}
    if needs_reference:
        from .reference import MAP_8, REFERENCE_LIMIT, check_actor_parity, open_match, reference_toolchain

        with reference_toolchain(args.reference) as toolchain:
            if args.verify_upstream:
                check_actor_parity(args.reference, toolchain, policy, checkpoint)
                report["upstream_actor_parity"] = "passed"
            if args.engine in ("java", "both"):
                with open_match(toolchain, MAP_8, 8) as env:
                    report["java"] = _measure(
                        lambda seed, seat: _reference_episode(env, REFERENCE_LIMIT, policy, seed, seat),
                        seeds, REFERENCE_LIMIT,
                    )
    if args.engine in ("native", "both"):
        report["native"] = _measure(
            lambda seed, seat: _native_episode(policy, seed, seat), seeds, MAX_TICKS
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
