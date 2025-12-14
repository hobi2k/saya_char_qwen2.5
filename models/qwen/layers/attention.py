from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .rotary import apply_rotary_pos_emb

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA/MQA 지원을 위해 K/V head를 Q head 개수에 맞추는 함수.

    입력:
        hidden_states: (B, Hkv, S, D)
        n_rep: Hq / Hkv (정수여야 함)

    출력:
        (B, Hkv * n_rep, S, D) == (B, Hq, S, D)

    직관:
        - GQA에서는 여러 Q head가 같은 K/V head를 공유한다.
        - 구현에서는 공유를 "view 상의 공유"로 만들기 어렵고,
          계산 편의를 위해 K/V를 head 차원으로 반복시켜 놓는다.

    주의:
        torch.repeat_interleave(dim=1)와 논리적으로 동일하지만,
        view/expand를 이용해 좀 더 효율적으로 만들었다.
    """
    B, Hkv, S, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # (B, Hkv, 1, S, D)로 만든 뒤 n_rep만큼 expand
    hidden_states = hidden_states[:, :, None, :, :].expand(B, Hkv, n_rep, S, D)
    # head 축을 (Hkv*n_rep)로 합침
    return hidden_states.reshape(B, Hkv * n_rep, S, D)


def eager_attention_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout_p: float,
    training: bool,
    num_key_value_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention 구현

    Args:
        q: (B, Hq, S, D)
        k: (B, Hkv, S, D)
        v: (B, Hkv, S, D)
        attention_mask:
            보통 (B, 1, S, S_k) 또는 (B, Hq, S, S_k) 형태.
            HF의 create_causal_mask 류는 "더하기 mask"로 들어오며,
            mask 위치에 매우 작은 값(예: -inf 또는 큰 음수)이 들어있다.
        scaling:
            1/sqrt(D)
        dropout_p:
            attention dropout 확률
        training:
            드롭아웃 적용 여부
        num_key_value_groups:
            Hq / Hkv. (GQA에서 q head들이 kv head를 공유하는 그룹 수)

    Returns:
        attn_output: (B, Hq, S, D)
        attn_weights: (B, Hq, S, S_k)
    """
    # 1. GQA: K/V head를 Q head에 맞춘다
    # k,v: (B, Hkv, S, D) -> (B, Hq, S, D)
    k = repeat_kv(k, num_key_value_groups)
    v = repeat_kv(v, num_key_value_groups)

    # 이제 q,k,v 모두 head 수가 같아져야 한다.
    # q: (B, Hq, S, D)
    # k: (B, Hq, S, D)
    # v: (B, Hq, S, D)

    # 2. Attention scores 계산
    # scores = Q @ K^T

    # q: (B, H, S, D)
    # k.transpose(-2, -1): (B, H, D, S)
    # matmul 결과: (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    # 3. Mask 적용
    # attention_mask는 "더하기(additive)" mask다.
    # 즉, 가려야 할 위치에 -inf(또는 매우 큰 음수)가 있어서
    # scores + mask 후 softmax를 하면 그 위치 확률이 0이 된다.
    if attention_mask is not None:
        # HF 구현에서는 가끔 scores의 key length(S_k)에 맞게 슬라이스한다.
        # 여기서는 k 길이가 S라고 가정하지만, cache가 있으면 S_k가 달라질 수 있다.
        # 안전하게 마지막 축 길이에 맞춰 슬라이스:
        S_q = q.shape[-2]
        S_k = k.shape[-2]
        scores = scores + attention_mask[..., :S_q, :S_k]

    # 4. Softmax (수치 안정성 포인트)
    # softmax는 fp16/bf16에서 불안정할 수 있으므로,
    # HF 구현처럼 float32로 softmax 한 뒤 다시 원래 dtype으로 되돌린다.
    attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # 5. Dropout (학습 시에만)
    if training and dropout_p > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)

    # 6. Attention output 계산
    # out = attn_weights @ V

    # attn_weights: (B, H, S, S_k) (보통 S_k == S)
    # v: (B, H, S_k, D)
    # 결과: (B, H, S, D)
    attn_output = torch.matmul(attn_weights, v)

    return attn_output, attn_weights

class Qwen2Attention(nn.Module):
    """
    Qwen2 Attention

    이 STEP에서는 '점수 계산(softmax)' 이전 단계까지:
      1. q/k/v projection
      2. head로 reshape + transpose
      3. RoPE 적용
      4. cache update 위치 확인
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

    # 1. Projection: hidden -> q/k/v
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

    # 2. Split heads: (B,S,heads*D) -> (B,heads,S,D)
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

    # 3. Apply RoPE to q, k (NOT v)
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

    # forward
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
        Qwen2 Attention forward (KV cache 포함, 학습용 완전 흐름)

        이 함수는 다음 질문에 코드로 답해야 한다:
        - 왜 K/V만 cache하는가?
        - RoPE는 cache 전/후 어디에 적용되는가?
        - generate()는 이 결과를 어떻게 재사용하는가?
        """

        # --------------------------------------------------
        # 입력
        # hidden_states: (B, S, hidden_size)
        #   - 프리필: S = prompt_len
        #   - 생성:   S = 1
        # --------------------------------------------------
        B, S, _ = hidden_states.shape

        # ==================================================
        # 1. Q / K / V projection
        # ==================================================
        # Q: (B, S, Q_heads * D)
        # K: (B, S, KV_heads * D)
        # V: (B, S, KV_heads * D)
        q_raw, k_raw, v_raw = self.project_qkv(hidden_states)

        # ==================================================
        # 2. split heads
        # ==================================================
        # attention 계산을 위해 (B, heads, S, D) 형태로 변환
        q = self.split_heads(q_raw, self.num_attention_heads)
        k = self.split_heads(k_raw, self.num_key_value_heads)
        v = self.split_heads(v_raw, self.num_key_value_heads)

        # ==================================================
        # 3. RoPE 적용 (q, k만)
        # ==================================================
        # RoPE는 "이 토큰이 시퀀스에서 어디에 위치하는가"를
        # attention score(QK^T)에 반영하기 위한 회전이다.
        #
        # 중요한 규칙:
        #   - cache에 저장되는 K는 반드시 RoPE가 적용된 상태여야 한다.
        cos, sin = position_embeddings
        q, k = self.apply_rope(q, k, position_embeddings)

        # ==================================================
        # 4. KV cache update (핵심)
        # ==================================================
        # generate() 단계에서는:
        #   - q: 항상 현재 step의 토큰만 포함 (S=1)
        #   - k, v: 과거 전체 + 현재 토큰
        #
        # 따라서:
        #   - q는 cache하지 않는다
        #   - k, v만 cache에 누적한다
        if past_key_values is not None:
            # HuggingFace cache는 다음 정보를 기대한다.
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }

            # update()는:
            #   - k, v를 layer_idx 기준으로 누적
            #   - 누적된 전체 k, v를 반환
            k, v = past_key_values.update(
                k,
                v,
                self.layer_idx,
                cache_kwargs,
            )

        # ==================================================
        # 5. Scaled Dot-Product Attention
        # ==================================================
        # 이 시점의 shape:
        #   q: (B, Hq, S, D)
        #   k: (B, Hkv, T_total, D)
        #   v: (B, Hkv, T_total, D)
        #
        # eager_attention_forward 내부에서:
        #   - GQA를 위해 k/v repeat
        #   - mask 적용
        #   - softmax(QK^T)
        #   - attention @ V
        attn_output, attn_weight = eager_attention_forward(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
            num_key_value_groups=self.num_key_value_groups,
        )

        # ==================================================
        # 6. heads merge
        # ==================================================
        # (B, Hq, S, D) -> (B, S, Hq*D)
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(B, S, self.num_attention_heads * self.head_dim)
        )

        # ==================================================
        # 7. output projection
        # ==================================================
        # 최종 output: (B, S, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weight