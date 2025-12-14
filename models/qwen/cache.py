from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache


# ============================================================
# LayerKV
# ============================================================
# Transformer는 "레이어마다" 서로 다른 Attention을 수행한다.
# 따라서 KV cache도 반드시 레이어 단위로 분리되어야 한다.
#
# 이 dataclass는
#   "한 개의 Transformer layer에서 누적된 K와 V"
# 를 명확히 표현하기 위한 구조체다.
#
# key/value shape:
#   (B, H, T, D)
#   B: batch size
#   H: attention heads (또는 kv heads)
#   T: 지금까지 처리한 전체 토큰 길이
#   D: head_dim
# ============================================================

@dataclass
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


# ============================================================
# HFCompatibleKVCache
# ============================================================
# 이 클래스의 목적은 단 하나다.
#
#   "HuggingFace generate()가 기대하는 KV cache 계약을
#    최소한의 코드로 충족시키는 것"
#
# 이 캐시는:
#   - K/V를 레이어별로 저장한다
#   - 새로운 토큰이 들어오면 뒤에 붙인다 (append)
#   - 현재까지의 전체 시퀀스 길이를 추적한다
#
# ⚠️ 주의:
#   - 이 캐시는 성능 최적화용이 아니다
#   - static cache, sliding window, rope 재정렬은 일부러 구현하지 않는다
#   - 목적은 "구조 이해"와 "HF 호환성"이다
# ============================================================

class HFCompatibleKVCache(Cache):
    def __init__(self):
        """
        Cache 초기화.

        _cache:
            Dict[layer_idx, LayerKV]
            Transformer의 각 layer마다
            누적된 (key, value)를 저장한다.

        _seq_length:
            지금까지 cache에 저장된 전체 토큰 수 T.
            HuggingFace는 이 값을 기반으로
            다음 토큰의 position_ids를 계산한다.
        """
        super().__init__()
        self._cache: Dict[int, LayerKV] = {}
        self._seq_length: int = 0

    # --------------------------------------------------------
    # get_seq_length
    # --------------------------------------------------------
    # HuggingFace generate() 내부에서 반드시 호출되는 메서드.
    #
    # generate()는 대략 이런 식으로 생각한다:
    #
    #   past_len = past_key_values.get_seq_length()
    #   position_ids = arange(past_len, past_len + new_tokens)
    #
    # 즉:
    #   이 함수는 "지금까지 몇 개의 토큰이 처리되었는가?"
    #   를 알려주는 유일한 정보원이다.
    # --------------------------------------------------------
    def get_seq_length(self) -> int:
        return self._seq_length

    # --------------------------------------------------------
    # update
    # --------------------------------------------------------
    # 이 메서드는 Attention layer 내부에서 호출된다.
    #
    # 호출 시점:
    #   - q/k/v projection 완료
    #   - RoPE가 q,k에 이미 적용된 상태
    #
    # 역할:
    #   - 새로 계산된 k,v를
    #     기존 cache 뒤에 이어 붙인다
    #   - 누적된 전체 k,v를 반환한다
    #
    # HuggingFace는 이 메서드의 "시그니처"를 강하게 가정한다.
    # 이름, 인자 순서가 바뀌면 generate()가 깨진다.
    # --------------------------------------------------------
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            key:
                새로 계산된 key.
                shape = (B, H, S_new, D)

                - 프리필(prompt) 단계:
                    S_new = prompt_length
                - 생성 단계:
                    S_new = 1

            value:
                새로 계산된 value.
                shape = (B, H, S_new, D)

            layer_idx:
                이 key/value가 어느 Transformer layer에 속하는지.
                반드시 layer-wise로 분리 저장해야 한다.

            cache_kwargs:
                HuggingFace cache 확장을 위한 정보.
                RoPE 모델에서는 보통 다음을 포함한다:
                    {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    }

                ⚠️ 이 최소 구현에서는
                - cache_kwargs를 "받기만" 하고
                - 내부적으로는 사용하지 않는다.
                (계약 충족 목적)

        Returns:
            key_all, value_all:
                지금까지 누적된 전체 key/value.
                shape = (B, H, T_total, D)
        """

        # --------------------------------------------
        # 첫 토큰(또는 프리필) 처리
        # --------------------------------------------
        if layer_idx not in self._cache:
            # 아직 이 layer에 대한 cache가 없다면
            # 그대로 저장한다.
            self._cache[layer_idx] = LayerKV(
                key=key,
                value=value,
            )
        else:
            # ----------------------------------------
            # 생성 단계: 기존 cache 뒤에 붙이기
            # ----------------------------------------
            old = self._cache[layer_idx]

            # sequence 차원(dim=-2)을 기준으로 이어 붙인다.
            # 이유:
            #   (B, H, T, D) 구조에서
            #   T가 "시간/토큰 축"이기 때문
            key = torch.cat([old.key, key], dim=-2)
            value = torch.cat([old.value, value], dim=-2)

            self._cache[layer_idx] = LayerKV(
                key=key,
                value=value,
            )

        # --------------------------------------------
        # 전체 시퀀스 길이 갱신
        # --------------------------------------------
        # 모든 layer는 동일한 T를 가진다고 가정한다.
        # (Transformer 구조상 항상 참)
        self._seq_length = key.shape[-2]

        return key, value
