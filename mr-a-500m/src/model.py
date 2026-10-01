"""Mr.A-500M model definition.

Decoder-only transformer matching the V20.0_STEP_1 architecture spec
(mra_500m_architecture_v1.json):

  - 26 layers, hidden 1280, 20 Q heads / 5 KV heads (GQA), head_dim 64
  - SwiGLU FFN (hidden 3584), RMSNorm (eps 1e-5), RoPE, context 4096
  - vocab 32000, tied word embeddings, random init, no pretrained weights

Total parameters: ~505.35M.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 1280)
        self.num_layers = kwargs.get("num_layers", 26)
        self.num_attention_heads = kwargs.get("num_attention_heads", 20)
        self.num_kv_heads = kwargs.get("num_kv_heads", 5)
        self.head_dim = kwargs.get("head_dim", 64)
        self.ffn_hidden_size = kwargs.get("ffn_hidden_size", 3584)
        self.context_length = kwargs.get("context_length", 4096)
        self.norm_eps = kwargs.get("norm_eps", 1e-5)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def precompute_rope_cos_sin(seq_len: int, head_dim: int, theta: float = 10000.0,
                            device=None, dtype=torch.float32):
    """Return (cos, sin) shaped (seq_len, head_dim)."""
    assert head_dim % 2 == 0
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    outer = torch.outer(pos, freqs)  # (seq_len, head_dim/2)
    emb = torch.cat([outer, outer], dim=-1)
    return torch.cos(emb), torch.sin(emb)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    cos = cos[: x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[: x.shape[2], :].unsqueeze(0).unsqueeze(0).to(x.dtype)
    return x * cos + rot * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, f = cfg.hidden_size, cfg.ffn_hidden_size
        self.gate_proj = nn.Linear(h, f, bias=False)
        self.up_proj = nn.Linear(h, f, bias=False)
        self.down_proj = nn.Linear(f, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MrAModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # weight tying
        # cache RoPE tables for the full context length
        cos, sin = precompute_rope_cos_sin(cfg.context_length, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.cfg.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        # idx: (B, T) token ids, T <= context_length
        B, T = idx.shape
        assert T <= self.cfg.context_length, f"seq {T} > context {self.cfg.context_length}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for blk in self.blocks:
            if use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)  # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length :]
            logits = self(ctx)[0, -1, :] / max(temperature, 1e-6)
            if top_k and top_k < logits.numel():
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = float("-inf")
            if top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(s, dim=-1), dim=-1)
                mask = cum - torch.softmax(s, dim=-1) > top_p
                s[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(0, si, s)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
