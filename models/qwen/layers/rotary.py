from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

@dataclass(frozen=True)
class RoteryOutput:
    """
    RoPE에서 생성되는 결과물(= position embedding의 실체).

    주의:
    - 여기서 "embedding"은 학습되는 lookup-table이 아니라,
      매 토큰 위치에 대해 계산된 cos/sin 행렬을 의미한다.
    """
    cos: torch.Tensor # shape: (B, S, D)
    sin: torch.Tensor # shape: (B, S, D)

class Qwen2RotaryEmbedding(nn.Module):
    """
    Qwen2 Rotary Embedding (RoPE) generator.

    핵심 역할:
    - query/key에 RoPE를 적용하기 위해 필요한 cos/sin 값을 만든다.

    입력:
    - x: hidden states (dtype/device 참조용)
    - position_ids: 각 토큰의 position 인덱스

    출력:
    - cos, sin: shape이 (batch, seq_len, head_dim) 인 텐서 2개

    왜 head_dim 기준인가?
    - RoPE는 attention head마다 적용된다.
    - query/key의 마지막 차원은 head_dim이므로,
      RoPE도 head_dim에 대해 cos/sin을 만든다.
    """
    inv_freq: torch.Tensor  # register_buffer 타입 힌트

    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()

        # max_position_embeddings:
        # - "이론상 지원 가능한 최대 토큰 길이"
        # - 기본 RoPE는 이 값을 반드시 쓰진 않지만,
        #   일부 RoPE 변형(dynamic rope 등)은 캐싱/스케일링에 활용한다.
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # rope_theta(base):
        # - RoPE 주파수의 base 값
        # - 논문/구현에서 흔히 10000을 사용하지만, 모델마다 다를 수 있음
        self.rope_theta = float(config.rope_parameters["rope_theta"])

        # head_dim (방어적 초기화):
        # - attention head 하나의 차원
        # - 보통 hidden_size / num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Qwen2 HF 구현은 rope_type에 따라 init 함수를 바꿀 수 있으나,
        # 지금 STEP에서는 "default rope"를 학습 목적상 정확히 해부한다.
        # (변형 RoPE는 이후 STEP에서 다룬다.)
        inv_freq, attention_scaling = self._compute_default_inv_freq(
            base=self.rope_theta,
            dim=self.head_dim,
            device=device,
        )

        # attention_scaling:
        # - default rope에서는 사실상 1.0 (스케일링 없음)
        # - 변형 rope (yarn, dynamic 등)에서 cos/sin에 곱해지는 스케일로 쓰인다.
        self.attention_scaling = float(attention_scaling)

        # inv_freq는 학습 파라미터가 아니라 "상수 테이블"에 가까움.
        # persistent=False:
        # - state_dict 저장에서 제외
        # - config 기반으로 재생성 가능하기 때문
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 원본 inv_freq를 별도 저장(변형 RoPE에서 업데이트/복원에 사용 가능)
        self.original_inv_freq = inv_freq

    # Default RoPE의 inv_freq 계산
    @staticmethod
    def _compute_default_inv_freq(
        base: float,
        dim: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Default RoPE에서 사용하는 inverse frequency(inv_freq)를 생성한다.

        RoPE의 핵심 주파수 정의:
        - inv_freq[i] = 1 / (base ** (2i / dim))
        (구현에서는 i를 "짝수 차원 인덱스"로 취급하므로 0,2,4,...에 대응)

        여기서 dim은 head_dim.

        반환:
            inv_freq: shape (dim/2,)  (짝수 인덱스 개수만큼)
            attention_factor: default에서는 1.0
        """
        # dim의 절반만 사용하는 이유:
        # - RoPE는 2차원 쌍(짝/홀)을 한 쌍으로 회전시킨다.
        # - 예: (d0,d1), (d2,d3), ...
        # - 따라서 주파수는 "쌍 개수" = dim/2 만큼만 있으면 된다.
        idx = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()  # (dim/2,)
        inv_freq = 1.0 / (base ** (idx / dim))  # (dim/2,)

        attention_factor = 1.0  # default RoPE에서는 스케일링을 하지 않음
        return inv_freq, attention_factor

    # cos/sin 생성
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> RotaryOutput:
        """
        주어진 position_ids에 대해 RoPE의 cos/sin을 생성한다.

        Args:
            x:
                실제 attention 계산에 쓰일 hidden tensor.
                여기서는 "dtype/device" 참조용으로만 사용된다.
                (cos/sin을 x와 같은 device, 적절한 dtype으로 맞추기 위함)

            position_ids:
                shape: (B, S)
                각 배치/토큰의 절대 position.
                - 캐시가 있으면 0..S-1이 아닐 수도 있음 (여기서 KV cache는 “이전에 계산한 key / value를 저장해 두고, 다음 토큰 생성 때 재사용하는 것”이다.)
                - 예: past length가 128이면 다음 토큰 position은 128부터 시작

        Returns:
            RotaryOutput(cos, sin)
                cos/sin shape: (B, S, D) where D = head_dim
        """
        # inv_freq: (D/2,)
        # 이를 (B, D/2, 1)로 확장해 position과 곱하기 쉽게 만든다.
        # - None 추가로 차원 확장
        # - expand로 batch 크기만큼 broadcast
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        # shape: (B, D/2, 1)

        # position_ids: (B, S)
        # 이를 (B, 1, S)로 바꿔 inv_freq와 matmul 가능하게 한다.
        position_ids_expanded = position_ids[:, None, :].float()
        # shape: (B, 1, S)

        # dtype 관련:
        # - cos/sin 계산은 수치적으로 민감할 수 있다.
        # - 원문(HF) 구현은 float32에서 cos/sin을 만든 뒤, x.dtype으로 되돌린다.
        # - 이유: fp16/bf16에서 cos/sin이 누적 오차/언더플로를 일으킬 수 있음
        # - 따라서 여기서도 float32로 강제한다.
        device = x.device

        inv_freq_expanded = inv_freq_expanded.to(device=device, dtype=torch.float32)
        position_ids_expanded = position_ids_expanded.to(device=device, dtype=torch.float32)

        # freqs = inv_freq @ position_ids

        # inv_freq_expanded: (B, D/2, 1)
        # position_ids_expanded: (B, 1, S)
        # matmul 결과: (B, D/2, S)
        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded)  # (B, D/2, S)

        # transpose: (B, S, D/2)
        freqs = freqs.transpose(1, 2)

        # emb를 (B, S, D)로 만들기 위해 freqs를 두 번 이어붙인다.
        # 이유:
        # - RoPE는 (d0,d1), (d2,d3) ... 쌍 단위 회전
        # - cos/sin을 적용할 때 q,k의 마지막 차원 D와 맞춰야 한다.
        # - 각 쌍에 동일한 각도(theta)를 적용하기 위해 freqs를 복제한다.
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, S, D)

        # cos/sin 계산 + scaling
        cos = emb.cos() * self.attention_scaling  # (B, S, D)
        sin = emb.sin() * self.attention_scaling  # (B, S, D)

        # 최종 dtype은 x와 동일하게 맞춘다.
        # - attention에서 q,k는 보통 bf16/fp16로 계산
        # - cos/sin도 동일 dtype이면 broadcast 시 효율적
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return RotaryOutput(cos=cos, sin=sin)
    
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE에서 사용하는 핵심 보조 함수.

    입력 x의 마지막 차원(D=head_dim)을 "절반씩" 나눠서,
    다음 변환을 수행한다:

        (x1, x2) -> (-x2, x1)

    여기서 x1은 앞쪽 D/2 차원, x2는 뒤쪽 D/2 차원이다.

    왜 이렇게 하냐?
    - RoPE는 head_dim을 2차원 쌍으로 보고 회전시킨다.
    - 2D 회전 분해식에서 필요한 항이 (-y, x)인데,
      전체 벡터에 대해 이 연산을 한 번에 적용하기 위함이다.

    Args:
        x: shape (..., D)
           D는 짝수여야 한다. (RoPE는 쌍 단위 회전이므로)

    Returns:
        shape (..., D)
    """

    # x[..., :D/2] -> x1
    # x[..., D/2:] -> x2
    # 최종: [-x2, x1]
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"rotate_half expects even last dim, got {d}")

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Position Embedding(RoPE)를 q, k에 적용한다.

    수식 관점:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

    중요한 shape 조건:
        q, k: (B, H, S, D)
        cos,sin: (B, S, D)

    cos/sin을 q/k에 곱하려면, H 차원에 대해 브로드캐스트가 필요하므로
        cos -> (B, 1, S, D)
        sin -> (B, 1, S, D)
    로 바꾸기 위해 unsqueeze_dim=1을 사용한다.

    Args:
        q: query tensor, shape (B, H, S, D)
        k: key tensor, shape (B, H(or H_kv), S, D)
        cos: cosine tensor, shape (B, S, D)
        sin: sine tensor, shape (B, S, D)
        unsqueeze_dim:
            cos/sin에 새로운 차원을 삽입할 위치.
            - q/k가 (B, H, S, D)이면 H 위치가 dim=1이므로 1을 넣는다.
            - 만약 구현을 바꿔 q/k가 (B, S, H, D)이면 unsqueeze_dim=2를 써야 한다.

    Returns:
        (q_rot, k_rot)  각 shape는 입력 q/k와 동일
    """
    # cos: (B, S, D) -> (B, 1, S, D)  [unsqueeze_dim=1]
    # 이렇게 하면 q: (B, H, S, D)와 곱할 때 H 차원으로 자동 broadcast 가능.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # rotate_half(q): (B, H, S, D) -> (B, H, S, D)
    # 그 다음 sin과 곱해지고 q*cos와 더해지면서 회전 효과를 만든다.
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot

def debug_rope_shapes(q, cos, unsqueeze_dim=1):
    """
    q shape를 보고 cos를 unsqueeze했을 때 broadcast가 되는지 확인한다.
    """
    print("q:", tuple(q.shape))
    print("cos:", tuple(cos.shape))
    cos_u = cos.unsqueeze(unsqueeze_dim)
    print("cos.unsqueeze:", tuple(cos_u.shape))
    # broadcast 가능 여부는 실제 곱을 시도해보면 확실하다.
    try:
        _ = q * cos_u
        print("Broadcast OK: q * cos.unsqueeze works")
    except RuntimeError as e:
        print("Broadcast FAILED:", e)