"""Train Mr.A-500M on the V19.2 token binaries.

T4-optimized (14.56 GiB VRAM):
  - BF16 mixed precision (no GradScaler needed)
  - gradient checkpointing
  - micro-batch size 1 + gradient accumulation
  - AdamW

Usage:
    python train.py --config config_500m.json --train-bin <path> --val-bin <path>
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import ModelConfig, MrAModel, count_parameters
from data import BinTokenReader, get_batch, estimate_val_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_500m.json")
    p.add_argument("--train-bin", default=None, help="override train binary path")
    p.add_argument("--val-bin", default=None, help="override val binary path")
    p.add_argument("--out-dir", default=None, help="override checkpoint dir")
    p.add_argument("--resume", default=None, help="resume from checkpoint (.pt)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def save_ckpt(path, model, opt, sched, epoch, opt_step, micro_step, rng_states, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "epoch": epoch,
        "opt_step": opt_step,
        "micro_step": micro_step,
        "rng": rng_states,
        "config": cfg,
    }, path)


def load_ckpt(path, model, opt, sched):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if sched and ck.get("scheduler"):
        sched.load_state_dict(ck["scheduler"])
    return ck


def get_lr(step, warmup, total, base, min_ratio):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    if step >= total:
        return base * min_ratio
    prog = (step - warmup) / max(total - warmup, 1)
    return base * min_ratio + 0.5 * base * (1 - min_ratio) * (1 + math.cos(math.pi * prog))


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    mcfg, tcfg, dcfg = cfg["model"], cfg["training"], cfg["data"]
    train_bin = args.train_bin or dcfg["train_binary"]
    val_bin = args.val_bin or dcfg["val_binary"]
    out_dir = args.out_dir or tcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and tcfg.get("precision", "bf16") == "bf16"

    model = MrAModel(ModelConfig(**mcfg)).to(device)
    n_params = count_parameters(model)
    print(f"parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            betas=tuple(tcfg["betas"]),
                            weight_decay=tcfg["weight_decay"])
    seq, micro_bs = mcfg["context_length"], tcfg["micro_batch_size"]
    accum = tcfg["grad_accum"]
    epochs = tcfg["epochs"]

    train_reader = BinTokenReader(train_bin)
    val_reader = BinTokenReader(val_bin)
    print(f"train tokens: {len(train_reader):,} | val tokens: {len(val_reader):,}")

    micro_per_epoch = len(train_reader) // (seq * micro_bs)
    total_opt_steps = epochs * micro_per_epoch // accum
    warmup = tcfg["warmup_steps"]
    print(f"micro-steps/epoch: {micro_per_epoch:,} | total opt steps: {total_opt_steps:,}")

    sched = None  # manual LR schedule via get_lr
    start_epoch, opt_step, micro_done = 0, 0, 0
    if args.resume:
        ck = load_ckpt(args.resume, model, opt, sched)
        start_epoch, opt_step = ck["epoch"], ck["opt_step"]
        micro_done = ck["micro_step"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        rng = np.random.default_rng()
        rng.bit_generator.state = ck["rng"]["np_rng"]
        print(f"resumed: epoch {start_epoch}, opt_step {opt_step}")

    log_path = os.path.join(out_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    def log(**kw):
        kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        logf.write(json.dumps(kw) + "\n")
        logf.flush()

    model.train()
    t0 = time.time()
    running_loss = 0.0
    for epoch in range(start_epoch, epochs):
        micro = micro_done if epoch == start_epoch else 0
        while micro < micro_per_epoch:
            x, y = get_batch(train_reader, micro_bs, seq, device, rng)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(x, use_checkpoint=tcfg["grad_checkpointing"])
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / accum).backward()
            running_loss += loss.item()

            if (micro + 1) % accum == 0:
                lr = get_lr(opt_step, warmup, total_opt_steps, tcfg["lr"], tcfg["min_lr_ratio"])
                for pg in opt.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                avg = running_loss / accum
                running_loss = 0.0
                if opt_step % tcfg["log_every"] == 0:
                    dt = time.time() - t0
                    toks = opt_step * seq * micro_bs * accum
                    print(f"ep {epoch} step {opt_step}/{total_opt_steps} "
                          f"loss {avg:.4f} lr {lr:.2e} tok {toks:,} {dt:.0f}s")
                    log(epoch=epoch, step=opt_step, loss=avg, lr=lr,
                        tokens_seen=toks, elapsed_s=dt)
                if opt_step % tcfg["eval_every"] == 0:
                    vl = estimate_val_loss(model, val_reader, seq, micro_bs,
                                           tcfg["eval_iters"], device,
                                           tcfg["grad_checkpointing"])
                    print(f"  [val] loss {vl:.4f}")
                    log(epoch=epoch, step=opt_step, val_loss=vl)
                if opt_step % tcfg["save_every"] == 0:
                    rng_states = {"torch": torch.get_rng_state(),
                                  "numpy": np.random.get_state(),
                                  "np_rng": rng.bit_generator.state}
                    save_ckpt(os.path.join(out_dir, f"ckpt_step{opt_step}.pt"),
                              model, opt, sched, epoch, opt_step, micro + 1,
                              rng_states, cfg)
                    print(f"  saved ckpt_step{opt_step}.pt")
            micro += 1
        # end of epoch
        vl = estimate_val_loss(model, val_reader, seq, micro_bs, tcfg["eval_iters"],
                               device, tcfg["grad_checkpointing"])
        rng_states = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                      "np_rng": rng.bit_generator.state}
        save_ckpt(os.path.join(out_dir, f"ckpt_epoch{epoch}.pt"),
                  model, opt, sched, epoch + 1, opt_step, 0, rng_states, cfg)
        print(f"epoch {epoch} done | val loss {vl:.4f}")
        log(epoch=epoch, step=opt_step, val_loss=vl, event="epoch_end")
        micro_done = 0

    logf.close()
    print("training complete")


if __name__ == "__main__":
    main()
