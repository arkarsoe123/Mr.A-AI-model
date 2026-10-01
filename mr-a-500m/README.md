# Mr.A-500M

Training code for **Mr.A-500M** — arkar's personal Myanmar+English conversational
model (~505.35M parameters, decoder-only transformer, trained from random
initialization).

## Architecture (V20.0_STEP_1 spec)

| | |
|---|---|
| Layers | 26 |
| Hidden size | 1280 |
| Attention | GQA — 20 Q heads / 5 KV heads, head dim 64 |
| FFN | SwiGLU, hidden 3584 |
| Norm | RMSNorm (eps 1e-5) |
| Positions | RoPE, context 4096 |
| Vocab | 32,000 (frozen SentencePiece tokenizer `mra_v10`) |
| Embeddings | tied |
| Init | random (no pretrained weights) |
| Params | 505,350,400 |

## Data

Frozen **V19.2** corpus (accepted 2026-09-21): uint16 little-endian token-id
binaries — 164,893,150 train tokens / 1,671,522 val tokens.

## Layout

```
mr-a-500m/
├── config_500m.json     # model + training hyperparameters
├── requirements.txt
├── src/
│   ├── model.py         # MrAModel (RMSNorm/RoPE/GQA/SwiGLU/tied embeddings)
│   ├── data.py          # uint16 memmap reader, random-chunk batching
│   ├── train.py         # BF16 + grad checkpointing + accum training loop
│   ├── generate.py      # sample from a checkpoint (needs mra_v10.model)
│   └── export_hf.py     # export to HF LLaMA format -> GGUF -> Android
└── notebooks/
    └── train_mra_500m_colab.ipynb
```

## Train on Colab T4

Open `notebooks/train_mra_500m_colab.ipynb` in Colab with a T4 GPU runtime.
Settings are T4-safe (14.56 GiB VRAM): BF16, gradient checkpointing,
micro-batch 1, grad-accum 64 → 262,144 tokens per optimizer step.

- 5 epochs ≈ 3,145 optimizer steps, ~30–45h of T4 total — run in multiple
  sessions, resuming from checkpoints:
  `python src/train.py --config config_500m.json --resume out/mra_500m/ckpt_epoch0.pt`
- Checkpoints, optimizer state and RNG state are all saved, so resume is exact.

## Export for Android

```bash
python src/export_hf.py --ckpt out/mra_500m/ckpt_epoch4.pt \
    --tokenizer /path/to/mra_v10.model --out hf_mra_500m
# then with llama.cpp:
python convert_hf_to_gguf.py hf_mra_500m --outfile mra-500m-f16.gguf
llama-quantize mra-500m-f16.gguf mra-500m-q4_k_m.gguf Q4_K_M   # ~300 MB
```

## Verification

All code is smoke-tested: model forward pass + parameter count (505.35M),
full training loop on synthetic data (including checkpoint resume),
`generate()` sampling, and the HF export round-trip.
