"""Run a short masked actor-critic training session with both seats active."""

import argparse

import numpy as np
import torch
from torch import nn

from micro_rts_rs import BatchedEnv
from micro_rts_rs._native import ACTIONS, BOARD_FEATURES, WIDTH


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(BOARD_FEATURES + 1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        self.actor = nn.Conv2d(16, ACTIONS, 1)
        self.critic = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1))

    def forward(self, observations):
        board = torch.as_tensor(observations["board"], dtype=torch.float32).flatten(0, 1)
        bank = torch.as_tensor(observations["credits"][..., 0], dtype=torch.float32).flatten()
        credits = (bank[:, None, None, None] / 20).expand(-1, WIDTH, WIDTH, -1)
        features = self.encoder(torch.cat((board / 12, credits), dim=-1).permute(0, 3, 1, 2))
        logits = self.actor(features).flatten(2).transpose(1, 2)
        mask = torch.as_tensor(observations["action_mask"], dtype=torch.bool).flatten(0, 1)
        return logits.masked_fill(~mask, -torch.inf), self.critic(features).flatten()


def train(steps, batch_size, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = BatchedEnv(batch_size, fog=True)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    observations = env.reset()
    finished = 0
    start = next(model.parameters()).detach().clone()
    for _ in range(steps):
        logits, values = model(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        next_views, rewards, terminated, truncated = env.step(actions.numpy().reshape(batch_size, 2, -1))
        with torch.no_grad():
            _, bootstrap = model(next_views)
            continues = torch.as_tensor(np.repeat(~terminated, 2))
            targets = torch.as_tensor(rewards.flatten(), dtype=torch.float32) + 0.99 * bootstrap * continues
        sources = torch.as_tensor(observations["source_mask"], dtype=torch.float32).flatten(0, 1)
        count = sources.sum().clamp_min(1)
        advantage = (targets - values).detach()
        actor_loss = -(distribution.log_prob(actions) * sources * advantage[:, None]).sum() / count
        entropy = (distribution.entropy() * sources).sum() / count
        loss = actor_loss + 0.5 * nn.functional.mse_loss(values, targets) - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for index in np.flatnonzero(terminated | truncated):
            reset = env.reset(int(index))
            for field, value in reset.items():
                next_views[field][index] = value
            finished += 1
        observations = next_views
    if not torch.isfinite(loss) or torch.equal(next(model.parameters()), start):
        raise RuntimeError("self-play training failed to update the model with a finite loss")
    print(f"Trained on {2 * steps * batch_size} player steps from both seats, "
          f"{finished} completed matches, final loss {loss.item():.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    train(args.steps, args.batch_size, args.seed)
