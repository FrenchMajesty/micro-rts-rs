# Relay, a Rust-native RTS benchmark

Relay is a deterministic 8×8 environment for reinforcement learning and
simultaneous-action lookahead. Each side opens with one base and one worker.
Players collect ore, train mobile units, and fight to eliminate the opposing
side. Occupying the central relay uncontested earns credits every eight ticks.

The published MicroRTS-Py GridNet checkpoint can play Relay through a policy
adapter. The evaluator also runs the same checkpoint on the pinned upstream Java
engine and its 8×8 benchmark map. These runs measure transfer between games.

## Run locally

Run these commands with Rust and Python 3.12 installed.

```sh
cargo test --workspace
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[policy,test]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/train_self_play.py
cargo run --release -p micro-rts-sim --example throughput
```

The policy extra supplies PyTorch. Rust, the PettingZoo interface, and the
batched environment can also run with the base install.

```python
from micro_rts_rs import RelayEnv, verify_replay

env = RelayEnv(fog=True, record_replay=True)
views, info = env.reset(seed=7)
while env.agents:
    joint_orders = {
        agent: env.action_space(agent).sample(mask=views[agent]["action_mask"])
        for agent in env.agents
    }
    views, rewards, terminated, truncated, info = env.step(joint_orders)

ticks, final_state_hash = verify_replay(env.replay_json())
```

The Rust entry points are `Game::new`, `reset`, `observe`, `legal_actions`,
and `step`. Use `Game::clone` for search from trusted full states. A `Replay`
records joint orders and a state hash after each tick. Use the player view and
legal masks as the inputs for fog agents. Replays are for trusted evaluation.

## Rules and Python interface

Each joint step advances one tick. Idle units accept one order each, and busy
units retain their pending orders. Unavailable orders and omitted orders yield
a one-tick wait on an idle unit. The completion snapshot determines occupancy,
attack targets, and resource availability. Destinations must be clear both
when issued and when completed. If multiple moves or spawns claim the same
cell on a tick, every claimant fails. If simultaneous gathers exceed a deposit's
remaining ore, every claimant fails. A depleted deposit opens its cell.

Training debits credits on acceptance and refunds failed spawns or interrupted
training. Completions take effect even when the acting unit is killed that tick.
Attacks hit the opposing occupant of the targeted cell in the completion
snapshot, including a unit moving away on that tick. After deaths and spawns,
relay income is paid and the match checks elimination. Mutual elimination is
terminal with zero reward. A match still running at tick 512 truncates.

The channels of `board[y, x, :]` are listed in storage order.

```text
wall, relay, visible, relative owner, kind, HP, ore or cargo,
pending action, ticks until completion
```

Relative owners are 0 for empty or neutral, 1 for self, and 2 for opponent.
Kinds are empty, ore, base, worker, and scout in that order from 0. Pending
action codes start at 0 for idle, followed by move, gather, deposit, train
worker, train scout, and attack. `credits` lists own credits and then opposing
credits, with `-1` for hidden opposing credits. In fog, terrain and relay
locations are public, and every friendly unit reveals cells within Manhattan
distance 3. The simulator itself is deterministic. Reset seeds the Python
action spaces for reproducible sampling.

Each player submits 64 row-major integers. Choice 0 is wait. Choices 1–24
are six groups of four in the order move, gather, deposit, train worker, train
scout, and attack. Each group uses north, east, south, west. The source mask
marks idle owned units. `BatchedEnv(n)` accepts `(n, 2, 64)` joint actions and
returns observations with leading `(n, 2)` dimensions. Call `reset(index)` to
reset a finished slot before the next batch step.

## Evaluate the published checkpoint

The adapter uses the fully observed 29-channel profile from MicroRTS-Py
`91ae6f54c92d5f97abb34bcaff3013adc51cb687`. It projects the native unit and
action vocabulary into GridNet, with scout mapped to Light. Relay ownership
and player credits have no GridNet input channels. Actor inference runs on
8×8 boards. The 16×16 value head is loaded for checkpoint validation.

For native matches, pass the path to the published checkpoint. [agent_sota.pt](https://github.com/Farama-Foundation/MicroRTS-Py/blob/91ae6f54c92d5f97abb34bcaff3013adc51cb687/experiments/gym-microrts-static-files/agent_sota.pt)

```sh
.venv/bin/micro-rts-eval --engine native --checkpoint /path/to/agent_sota.pt
```

For both benchmarks, use a separate Python 3.9 environment and JDK 8. Set
`JAVA_HOME` to JDK 8 and put its `bin` directory first on `PATH`. The commands
below prepare the pinned reference checkout and build its Java jar.

```sh
mkdir -p .cache
git clone https://github.com/Farama-Foundation/MicroRTS-Py.git .cache/microrts-reference
git -C .cache/microrts-reference checkout --detach 91ae6f54c92d5f97abb34bcaff3013adc51cb687
git -C .cache/microrts-reference submodule update --init gym_microrts/microrts
(cd .cache/microrts-reference && bash build.sh)

uv venv --python 3.9 .cache/reference-env
uv pip install --python .cache/reference-env/bin/python -r requirements-reference.txt .
.cache/reference-env/bin/micro-rts-eval --engine both --reference .cache/microrts-reference --verify-upstream
```

Java runs in a separate headless process. The runner drives the pinned Java
self-play client and calls the upstream Python observation encoder. This
avoids loading the JVM into Python and reports a child-process failure as a
Python error. The parity check compares logits and masked probabilities on
16×16 Java observations before evaluation on the upstream 8×8 map.

Evaluation plays the checkpoint against a random legal-action opponent on
both sides for each seed, with seeds 7 and 11 by default. The native horizon
is 512 ticks and the upstream reference horizon is 2,000. The report includes
per-match outcomes, truncations, episode lengths, and end-to-end throughput.

### Initial verification run

The 16×16 actor parity check passed. With seeds 7 and 11 on an Apple M3 Max,
the checkpoint produced the following four-match results against the random
opponent.

- Relay recorded 1 win and 3 timeout draws, averaged 412 ticks per match,
  and ran at about 2,530 joint ticks per second including CPU inference.
- Java MicroRTS recorded 1 win, 1 loss, and 2 timeout draws, averaged 1,774
  ticks per match, and ran at about 2,410 joint ticks per second including CPU
  inference and communication with the Java process.

The Rust-only release benchmark reached about 10.5 million joint ticks per
second on its fixed opening and idle-order workload. These timings describe
the verification run, with different workloads and tick limits across checks.

Upstream sources and attribution are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
