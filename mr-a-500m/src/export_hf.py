"""Export a trained Mr.A-500M checkpoint to HuggingFace LLaMA format.

Mr.A is architecturally LLaMA-compatible (RMSNorm, RoPE, SwiGLU, GQA), so the
export maps Mr.A tensor names onto the standard `LlamaForCausalLM` layout.
The result converts cleanly to GGUF with llama.cpp's convert_hf_to_gguf.py,
then quantizes to Q4_K_M (~300MB) for Android on-device inference.

Tied embeddings are materialized as a separate lm_head (GGUF/HF tooling
expects an explicit output matrix).

Usage:
    python export_hf.py --ckpt out/ckpt_epoch4.pt --tokenizer mra_v10.model \
        --out hf_mra_500m
"""

import argparse
import json
import os

import torch

from model import ModelConfig, MrAModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", default=None, help="mra_v10.model to copy alongside")
    p.add_argument("--config", default="config_500m.json")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.config) as f:
        mcfg = json.load(f)["model"]
    cfg = ModelConfig(**mcfg)

    model = MrAModel(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    sd = model.state_dict()

    hf = {}
    hf["model.embed_tokens.weight"] = sd["tok_emb.weight"].to(torch.float16)
    for i in range(cfg.num_layers):
        p = f"blocks.{i}."
        q = f"model.layers.{i}."
        hf[q + "self_attn.q_proj.weight"] = sd[p + "attn.q_proj.weight"].to(torch.float16)
        hf[q + "self_attn.k_proj.weight"] = sd[p + "attn.k_proj.weight"].to(torch.float16)
        hf[q + "self_attn.v_proj.weight"] = sd[p + "attn.v_proj.weight"].to(torch.float16)
        hf[q + "self_attn.o_proj.weight"] = sd[p + "attn.o_proj.weight"].to(torch.float16)
        hf[q + "mlp.gate_proj.weight"] = sd[p + "ffn.gate_proj.weight"].to(torch.float16)
        hf[q + "mlp.up_proj.weight"] = sd[p + "ffn.up_proj.weight"].to(torch.float16)
        hf[q + "mlp.down_proj.weight"] = sd[p + "ffn.down_proj.weight"].to(torch.float16)
        hf[q + "input_layernorm.weight"] = sd[p + "attn_norm.weight"].to(torch.float16)
        hf[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"].to(torch.float16)
    hf["model.norm.weight"] = sd["final_norm.weight"].to(torch.float16)
    # materialize a separate lm_head (untie for HF/GGUF tooling)
    hf["lm_head.weight"] = sd["lm_head.weight"].clone().to(torch.float16)

    try:
        from safetensors.torch import save_file
        save_file(hf, os.path.join(args.out, "model.safetensors"))
        print("wrote model.safetensors")
    except ImportError:
        torch.save(hf, os.path.join(args.out, "pytorch_model.bin"))
        print("wrote pytorch_model.bin (pip install safetensors for .safetensors)")

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.ffn_hidden_size,
        "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "max_position_embeddings": cfg.context_length,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    if args.tokenizer:
        import shutil
        shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.model"))
        print("copied tokenizer.model")

    print(f"\nexport complete -> {args.out}")
    print("next: python convert_hf_to_gguf.py {0} --outfile mra-500m-f16.gguf".format(args.out))
    print("then: llama-quantize mra-500m-f16.gguf mra-500m-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
