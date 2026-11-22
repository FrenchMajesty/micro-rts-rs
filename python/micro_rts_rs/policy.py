"""CPU inference for the published MicroRTS-Py GridNet checkpoint."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .gridnet import CHANNELS, GROUPS, MASK_WIDTH

PY_REVISION = "91ae6f54c92d5f97abb34bcaff3013adc51cb687"
JAVA_REVISION = "676e52d34098a050a0f432b7d523f245f1352ca6"
CHECKPOINT_PATH = "experiments/gym-microrts-static-files/agent_sota.pt"
CHECKPOINT_SHA256 = "3a3510bfbe873fc260b74cbc588b092a46cab8f34a32509c48fbc8c2535fc7bd"


class _ChannelsLast(nn.Module):
    def forward(self, cells):
        return cells.permute(0, 2, 3, 1)


class _ChannelsFirst(nn.Module):
    def forward(self, cells):
        return cells.permute(0, 3, 1, 2)


class GridNetPolicy(nn.Module):
    """GridNet architecture matching the pinned MIT-licensed experiment."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _ChannelsFirst(),
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, MASK_WIDTH, 3, stride=2, padding=1, output_padding=1),
            _ChannelsLast(),
        )
        # Register the value head for strict loading. Inference only calls the actor.
        self.critic = nn.Sequential(nn.Flatten(), nn.Linear(1024, 128), nn.ReLU(), nn.Linear(128, 1))
        self.register_buffer("mask_value", torch.tensor(-1e8))

    @classmethod
    def from_checkpoint(cls, path):
        path = Path(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != CHECKPOINT_SHA256:
            raise ValueError(f"expected the pinned GridNet checkpoint SHA-256; got {actual}")
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return model.eval()

    def forward(self, board):
        if board.ndim != 4 or board.shape[-1] != CHANNELS:
            raise ValueError("expected [batch, height, width, 29] GridNet observations")
        if board.shape[1] % 4 or board.shape[2] % 4:
            raise ValueError("GridNet board dimensions must be divisible by four")
        return self.actor(self.encoder(board.float()))

    def masked_probabilities(self, logits, factor_mask):
        if logits.shape != factor_mask.shape or logits.shape[-1] != MASK_WIDTH:
            raise ValueError("logits and masks must have matching GridNet action shapes")
        return tuple(
            torch.where(allowed.bool(), group, self.mask_value).softmax(dim=-1)
            for group, allowed in zip(logits.split(GROUPS, dim=-1), factor_mask.split(GROUPS, dim=-1))
        )

    @torch.inference_mode()
    def choose(self, inputs, generator):
        board = torch.from_numpy(inputs.board).unsqueeze(0)
        mask = torch.as_tensor(inputs.factor_mask, dtype=torch.bool)
        logits = self(board).reshape(-1, MASK_WIDTH)
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite GridNet logits")
        actions = np.zeros((len(inputs.source_mask), len(GROUPS)), dtype=np.int64)
        active = inputs.source_mask.astype(bool)
        if active.any():
            probabilities = self.masked_probabilities(logits[active], mask[active])
            actions[active] = torch.cat([
                torch.multinomial(group, 1, generator=generator) for group in probabilities
            ], dim=-1).numpy()
        return actions
