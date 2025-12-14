# models/qwen2/layers/attention.py

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .rotary import apply_rotary_pos_emb


class Qwen2Attention(nn.Module):
    """
    Qwen2 Attention (학습용 리팩토링 버전).

    이 STEP(3-1)에서는 '점수 계산(softmax)' 이전 단계까지:
      1) q/k/v projection
      2) head로 reshape + transpose
      3) RoPE 적용
      4) (선택) cache update 위치 확인

    까지만 정확히 이해하도록 만든다.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # head_dim:
        # - attention head 하나의 차원
        # - 보통 hidden_size / num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # GQA에서 중요한 값:
        # - num_attention_heads: Q head 개수
        # - num_key_value_heads: K/V head 개수
        # - num_key_value_groups = Q_heads / KV_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # scaling:
        # - dot-product attention에서 QK^T에 곱하는 1/sqrt(d)
        self.scaling = self.head_dim ** -0.5

        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # (가중치 호환 핵심) projection 계층 이름/shape는 절대 바꾸지 말 것.
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # Qwen2는 일부 layer에 sliding attention을 적용할 수 있다.
        # 이 STEP에서는 sliding 자체 구현은 다음 단계에서 다룬다.
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else "full_attention"
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    # -----------------------
    # 1) Projection: hidden -> q/k/v
    # -----------------------
    def project_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        입력 hidden_states로부터 Q/K/V를 projection 한다.

        Args:
            hidden_states: (B, S, hidden_size)

        Returns:
            q: (B, S, Q_heads * D)
            k: (B, S, KV_heads * D)
            v: (B, S, KV_heads * D)

        왜 q와 kv의 head 수가 다를 수 있나?
            - GQA/MQA 지원을 위해서다.
        """
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    # -----------------------
    # 2) Split heads: (B,S,heads*D) -> (B,heads,S,D)
    # -----------------------
    def split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        projection 결과를 head 단위로 쪼갠다.

        Args:
            x: (B, S, num_heads * D)
            num_heads: head 개수 (Q_heads 또는 KV_heads)

        Returns:
            x: (B, num_heads, S, D)

        구현 상세:
            - view로 마지막 차원을 (num_heads, D)로 쪼갬
            - transpose로 head 차원을 앞으로 가져옴
        """
        B, S, _ = x.shape

        # 여기서 -1은 num_heads*D여야 하고, D는 self.head_dim과 일치해야 한다.
        # view 이후 shape: (B, S, num_heads, D)
        x = x.view(B, S, num_heads, self.head_dim)

        # attention 계산은 보통 (B, heads, S, D) 형태가 편하므로 transpose
        x = x.transpose(1, 2).contiguous()
        return x

    # -----------------------
    # 3) Apply RoPE to q, k (NOT v)
    # -----------------------
    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k에 RoPE를 적용한다.

        Args:
            q: (B, Q_heads, S, D)
            k: (B, KV_heads, S, D) 또는 (B, Q_heads, S, D) (구현 흐름에 따라 다를 수 있음)
            position_embeddings: (cos, sin)
                cos/sin: (B, S, D)

        Returns:
            q_rot, k_rot: 입력과 동일 shape

        왜 v에는 RoPE를 적용하지 않나?
            - attention의 "어디를 볼지"는 QK^T로 결정된다.
            - 위치 정보는 score에만 들어가면 충분하며,
              V는 선택된 정보를 '가져오는 값'이기 때문에 회전이 필요 없다.
        """
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        return q, k

    # -----------------------
    # forward (STEP 3-1 범위)
    # -----------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        STEP 3-1 범위:
            - q/k/v projection
            - head split
            - RoPE 적용
            - (옵션) cache update 위치만 표시

        다음 STEP 3-2에서:
            - QK^T + mask + softmax + dropout + (attn @ V) 까지 구현한다.
        """

        # 입력: (B, S, hidden_size)
        B, S, H = hidden_states.shape

        # 1) projection
        q_raw, k_raw, v_raw = self.project_qkv(hidden_states)
        # q_raw: (B, S, Q_heads*D)
        # k_raw: (B, S, KV_heads*D)
        # v_raw: (B, S, KV_heads*D)

        # 2) split heads
        q = self.split_heads(q_raw, self.num_attention_heads)     # (B, Q_heads, S, D)
        k = self.split_heads(k_raw, self.num_key_value_heads)     # (B, KV_heads, S, D)
        v = self.split_heads(v_raw, self.num_key_value_heads)     # (B, KV_heads, S, D)

        # 3) RoPE 적용 (q,k만)
        q, k = self.apply_rope(q, k, position_embeddings)

        # 4) (옵션) KV cache update는 여기서 일어난다.
        # Qwen2 HF 구현에서는 past_key_values.update(...) 호출로
        # k,v가 누적 저장된다.
        if past_key_values is not None:
            # 여기서는 인터페이스 설명만 남긴다.
            # 실제 구현은 STEP 3-3 (KV cache)에서 다룬다.
            #
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            pass

        # STEP 3-1의 출력은 "attention 계산 직전의 q,k,v"라고 생각하면 된다.
        return q, k, v
