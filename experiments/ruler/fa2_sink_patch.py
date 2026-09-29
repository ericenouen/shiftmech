"""Run the EdenQiao checkpoints' sink attention on flash-attn instead of their bundled
Triton kernel, which does full quadratic work even in the sliding-window layers.

The learnable sink only adds exp(sink) to the softmax denominator, so
out = flash_attn_out * sigmoid(lse - sink), which is exact.
"""
import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input


def _fa2_sink_attention_forward(
    self, query_states, key_states, value_states, attention_mask,
    query_length, softmax_scale, causal, window_size=(-1, -1), cache_seqlens=None,
):
    """Drop-in for MiniCPMFlashAttention2._triton_attention_forward.

    Same signature, same return shape (batch, q_len, num_heads, head_dim).
    Note we deliberately do NOT repeat_interleave the KV heads for GQA the way the
    Triton path does -- flash-attn handles GQA natively and the repeat is pure cost.
    """
    bsz = query_states.shape[0]
    if softmax_scale is None:
        softmax_scale = 1.0 / (self.head_dim ** 0.5)
    sink = self.softmax_offset.to(torch.float32)          # (num_heads,)

    if attention_mask is not None:
        q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )
        cu_q, cu_k = cu_seq_lens
        max_q, max_k = max_seq_lens
        out, lse, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            dropout_p=0.0, softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, return_attn_probs=True,
        )
        # out (total_q, nheads, hd); lse (nheads, total_q)
        factor = torch.sigmoid(lse.transpose(0, 1).float() - sink.unsqueeze(0))
        out = (out.float() * factor.unsqueeze(-1)).to(out.dtype)
        return pad_input(out, indices_q, bsz, query_length)

    out, lse, _ = flash_attn_func(
        query_states, key_states, value_states,
        dropout_p=0.0, softmax_scale=softmax_scale, causal=causal,
        window_size=window_size, return_attn_probs=True,
    )
    # out (b, q, h, d); lse (b, h, q)
    factor = torch.sigmoid(lse.permute(0, 2, 1).float() - sink.view(1, 1, -1))
    return (out.float() * factor.unsqueeze(-1)).to(out.dtype)


def apply_fa2_sink_patch(model):
    """Patch every attention module of `model` in place. Returns how many."""
    n = 0
    seen = set()
    for layer in model.model.layers:
        cls = type(layer.self_attn)
        if cls.__name__ != "MiniCPMFlashAttention2":
            raise RuntimeError(
                f"expected MiniCPMFlashAttention2, got {cls.__name__}; "
                "load the model with attn_implementation='flash_attention_2'"
            )
        if cls not in seen:
            cls._triton_attention_forward = _fa2_sink_attention_forward
            seen.add(cls)
        n += 1
    return n
