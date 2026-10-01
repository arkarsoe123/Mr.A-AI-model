"""Generate text with a trained Mr.A-500M checkpoint.

Requires the frozen SentencePiece tokenizer mra_v10 (mra_v10.model).

Usage:
    python generate.py --ckpt out/ckpt_epoch4.pt --tokenizer mra_v10.model \
        --prompt "မင်္ဂလာပါ" --max-new-tokens 128
"""

import argparse
import json

import torch

from model import ModelConfig, MrAModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True, help="path to mra_v10.model")
    p.add_argument("--config", default="config_500m.json")
    p.add_argument("--prompt", default="မင်္ဂလာပါ")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def main():
    import sentencepiece as spm

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        mcfg = json.load(f)["model"]

    tok = spm.SentencePieceProcessor(model_file=args.tokenizer)
    model = MrAModel(ModelConfig(**mcfg)).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
        out = model.generate(ids, args.max_new_tokens, args.temperature,
                             args.top_k, args.top_p)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
