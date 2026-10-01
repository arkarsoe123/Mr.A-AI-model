"""Data loading for the V19.2 token binaries.

The corpus was built as uint16 little-endian token-id binaries:
  - train: 164,893,150 tokens
  - val:     1,671,522 tokens
Tokenized with the frozen SentencePiece tokenizer mra_v10 (vocab 32000).

Training uses random chunk sampling (nanoGPT style): each micro-batch is a
random contiguous slice of the binary. Validation iterates deterministically.
"""

import numpy as np
import torch


class BinTokenReader:
    """Memory-mapped reader over a uint16 little-endian token binary."""

    def __init__(self, path: str):
        self.path = path
        # '<u2' = little-endian uint16, matching the V19.2 STEP 3B build
        self.data = np.memmap(path, dtype="<u2", mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def slice(self, start: int, length: int) -> np.ndarray:
        return self.data[start: start + length]


def get_batch(reader: BinTokenReader, batch_size: int, seq_len: int,
              device: torch.device, rng: np.random.Generator):
    """Random contiguous chunks: x = ids[i:i+T], y = ids[i+1:i+T+1]."""
    n = len(reader)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    x = np.stack([reader.slice(int(s), seq_len) for s in starts])
    y = np.stack([reader.slice(int(s) + 1, seq_len) for s in starts])
    x = torch.from_numpy(x.astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(y.astype(np.int64)).to(device, non_blocking=True)
    return x, y


def get_val_batch(reader: BinTokenReader, batch_size: int, seq_len: int,
                  step: int, device: torch.device):
    """Deterministic sequential chunks for validation."""
    n = len(reader)
    total = n // seq_len
    starts = [((step * batch_size + b) % total) * seq_len for b in range(batch_size)]
    x = np.stack([reader.slice(s, seq_len) for s in starts])
    y = np.stack([reader.slice(s + 1, seq_len) for s in starts])
    x = torch.from_numpy(x.astype(np.int64)).to(device, non_blocking=True)
    y = torch.from_numpy(y.astype(np.int64)).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_val_loss(model, reader: BinTokenReader, seq_len: int, batch_size: int,
                      iters: int, device: torch.device, use_checkpoint: bool = False) -> float:
    model.eval()
    losses = []
    for i in range(iters):
        x, y = get_val_batch(reader, batch_size, seq_len, i, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(x, use_checkpoint=use_checkpoint)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))
