"""
Qwen2Tokenizer 구현 코드

이 파일은 Qwen2 계열 LLM이 사용하는 토크나이저를 파이썬으로 정의한 코드이다.
특히, HuggingFace의 tokenizers 라이브러리를 이용해서 BPE(Byte-Pair Encoding) 기반 토큰화 파이프라인을 구성하고,
이를 HuggingFace Transformers 쪽에서 공통으로 사용하는 TokenizersBackend 인터페이스에 맞게 래핑한다.


- 사용 예시
from saya_qwen2.config import SayaQwen2Config

# 공식 Qwen2-7B 구조 호환 모드(가중치 로딩을 염두)
cfg = SayaQwen2Config(
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=22016,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    max_position_embeddings=32768,
)
"""
from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any

# Transformers 버전에 따라 import 경로가 다를 수 있음
try:
    from transformers import PretrainedConfig
except:
    from transformers.configuration_utils import PretrainedConfig

# 1. RopeParameters를 "프로젝트 친화적"으로 단순화
#
# HF 내부의 RopeParameters 클래스를 그대로 끌고 오면
# - 내부 유틸 의존성이 늘고
# - 구현 단계에서 불필요한 복잡도가 생긴다.
#
# 그래서 내 프로젝트에서는 "RoPE 설정을 담는 dict 스펙"을 명시적으로 정의한다.
#
# rope_theta:
#   - RoPE의 base(θ) 값
#   - long-context 모델은 보통 큰 값을 사용 (예: 1e6)
#
# scaling:
#   - 긴 컨텍스트로 확장하기 위한 스케일링 관련 옵션(선택)
#   - 프로젝트마다 구현 방식이 달라질 수 있으므로 dict 형태로 열어둔다.
RopeScalingType = Literal["none", "linear", "dynamic", "yarn", "ntk"]

def _normalize_rope_parameters(
        rope_parameters: Optional[Dict[str, Any]],
        *,
        default_rope_theta: float,
) -> Dict[str, Any]:
    """
    rope_parameters를 "항상 dict" 형태로 정규화한다.

    RoPE란?
    - RoPE는 “위치 정보를 임베딩 벡터에 더하지 않고, Query / Key 벡터를 위치에 따라 회전시키는 방식”이다. (절대 위치가 아니라 상대 거리 기반 어텐션)

    왜 필요한가?
    - modeling 구현 시 config.rope_parameters를 항상 같은 형태로 쓰고 싶기 때문.
    - 사용자가 rope_parameters를 안 주면 기본값으로 채워 넣어야 함.

    반환 스펙(권장):
    {
        "rope_theta": float,
        "scaling": {
            "type": RopeScalingType,
            ... 추가 파라미터 ...
        }
    }
    """
    if rope_parameters is None:
        # 가장 기본 형태: theta만 둔다. scaling은 "none"으로 둔다.
        return {"rope_theta": float(default_rope_theta), "scaling":{"type":"none"}}
    
    # 사용자가 rope_theta를 명시하지 않았다면 기본값 보강
    if "rope_theta" not in rope_parameters:
        rope_parameters["rope_theta"] = float(default_rope_theta)
    else:
        rope_parameters["rope_theta"] = float(rope_parameters["rope_theta"])

    # scaling 섹션 정리
    scaling = rope_parameters.get("scaling", None)
    if scaling is None:
        rope_parameters["scaling"] = {"type":"none"}
    else:
        # scaling.type은 최소한 존재하도록 강제한다.
        scaling_type = scaling.get("type", "none")
        scaling["type"] = scaling_type
        rope_parameters["scaling"] = scaling

    return rope_parameters

# 2. layer_types 검증을 "프로젝트 내부 함수"로 구현
#
# HF 내부 layer_type_validation을 그대로 가져오면 내부 의존성 증가.
# 또한 초기 구현 단계에서는 "full_attention / sliding_attention" 정도만 있으면 충분하다.
#
# 나중에 attention 패턴을 추가하고 싶으면 allowed set만 늘리면 된다.
_ALLOWED_LAYER_TYPES = {"full_attention", "sliding_attention"}

def _validate_layer_type(layer_types: List[str], num_hidden_layers: int) -> None:
    """
    layer_types가 모델 레이어 수와 맞고, 값이 허용된 타입인지 검증한다.

    역할
    - 레이어마다 attention 방식을 바꾸는 구조는 구현이 유연하지만,
      config가 잘못되면 runtime에서 매우 늦게(학습/추론 도중) 터진다.
    - 따라서 config 초기화 시점에 즉시 검증하는 것이 좋다.
    """
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            f"layers_type length mismatch: got {len(layer_types)}"
            f"but num_hidden_layers={num_hidden_layers}"
        )
    unknown = [t for t in layer_types if t not in _ALLOWED_LAYER_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown layer_types={unknown}. Allowed={_ALLOWED_LAYER_TYPES}"
        )

# 3. 메인 Config 클래스
class SayaQwen2Config(PretrainedConfig):
    """
    SayaQwen2Config

    역할:
    1. modeling 구현 용이화 (config를 단일 진실로 사용)
    2. 실험 파라미터 한 곳에 집중
    3. 체크포인트/재현성 상승 (config.json으로 모든 설정 공유)

    규칙:
    - 공식 Qwen2 가중치를 그대로 불러오고 싶다면,
      아래 구조 파라미터(hidden_size, num_layers, head 수, mlp 차원 등)를
      공식 모델과 동일하게 유지해야 한다.
    """
    # HF AutoConfig가 모델을 식별하는 키
    model_type = "saya-qwen2"

    # generation 시 past_key_values는 내부 캐시라 출력에서 무시하는 경우가 많다.
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            # Tokenizer/Embedding 계약 파라미터
            vocab_size: int = 151_936,
            # vocab_size는 "토큰 ID의 최대값 + 1"이다.
            # embedding.weight의 shape는 [vocab_size, hidden_size]가 된다.
            # tokenizer가 생성하는 토큰 ID 범위가 여기와 반드시 맞아야 한다.

            # Transformer 기본 차원
            hidden_size: int = 4096,
            # hidden_size = d_model. 모든 레이어에서 흐르는 벡터 차원
            # attention head dim = hidden_size / num_attention_heads가 정수여야 한다.

            intermediate_size: int = 22016,
            # MLP(FFN)의 확장 차원.
            # Qwen 계열은 보통 SwiGLU 계열을 쓰며, hidden_size보다 훨씬 크다.

            num_hidden_layers: int = 32,
            # Transformer block(레이어) 개수

            # Attention 구조
            num_attention_heads: int = 32,
            # Query head 수

            num_key_value_heads: Optional[int] = 32,
            # KV head 수
            # - num_key_value_heads == num_attention_heads -> MHA
            # - num_key_value_heads == 1 -> MQA
            # - 그 중간 -> GQA
            #
            # 초기 구현 안정성을 위해, 처음에는 MHA(동일)로 고정

            attention_dropout: float = 0.0,
            # attention 확률에 적용하는 dropout

            hidden_act:str = "silu",
            # MLP activation
            # Qwen 계열은 silu를 많이 사용

            # Context / Positional Encoding
            max_position_embeddings: int = 32768,
            # 모델이 처리할 수 있다고 "가정하는" 최대 컨텍스트 길이
            # tokenizer.model_max_length도 일반적으로 여기에 맞춘다.

            rope_theta: float = 1_000_000.0,
            # RoPE base theta,
            # long-context를 염두에 두고 큰 값이 자주 사용된다.

            rope_parameters: Optional[Dict[str, Any]] = None,
            # 로프 스케일링까지 포함한 상세 설정
            # None이면 (rope_theta, scaling=none)로 자동 정규화된다.

            # Normalization / Init / Cacje
            rms_norm_eps: float = 1e-6,
            # RMSNorm epsilon. 수치 안정성
            initializer_range: float = 0.02,
            # weight 초기화 표준편차(보통 truncated normal)
            use_cache: bool = True,
            # generation에서 KV cache를 쓸지 여부

            tie_word_embeddings: bool = False,
            # input embedding과 output lm_head weight tying 여부
            # Qwen2 원본은 보통 False

            # Sliding Window Attention (선택 가능)
            use_sliding_window: bool = False,
            sliding_window: int = 4096,
            max_window_layers: int = 28,
            layer_types: Optional[List[str]] = None,
            # sliding window는 "긴 컨텍스트에서 메모리를 줄이기 위한" 기법.

            # layer_types를 명시하면 레이어별 attention 패턴을 강제한다.
            # None이면 num_hidden_layers에 맞춰 자동 생성한다.

            # 기타 (HF 호환을 위한 확장 슬롯)
            **kwargs: Any,
    ):
        # num_key_value_heads를 None으로 주는 경우도 있을 수 있으므로,
        # 기본은 num_attention_heads로 맞춘다(= MHA).
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # “shape를 결정하는 핵심 파라미터들” 저장
        # 이 값들이 바뀌면 대부분의 weight shape가 바뀐다.
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)

        # head_dim이 정수인지 즉시 검증
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads. "
                f"Got hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )

        # attention / activation
        self.attention_dropout = float(attention_dropout)
        self.hidden_act = str(hidden_act)

        # context / rope
        self.max_position_embeddings = int(max_position_embeddings)

        # rope 설정을 dict로 “항상 같은 형태”로 정규화한다.
        self.rope_parameters = _normalize_rope_parameters(
            rope_parameters,
            default_rope_theta=float(rope_theta),
        )

        # norm / init / cache
        self.rms_norm_eps = float(rms_norm_eps)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)

        # sliding window
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window) if self.use_sliding_window else None
        self.max_window_layers = int(max_window_layers)

        # layer_types 자동 생성 규칙:
        # - sliding_window를 쓰는 경우:
        #   * 앞쪽 max_window_layers는 full_attention
        #   * 그 이후는 sliding_attention
        # - sliding_window를 안 쓰는 경우: 전부 full_attention
        if layer_types is None:
            if self.sliding_window is None:
                self.layer_types = ["full_attention"] * self.num_hidden_layers
            else:
                self.layer_types = [
                    "sliding_attention" if i >= self.max_window_layers else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
        else:
            self.layer_types = list(layer_types)

        _validate_layer_type(self.layer_types, self.num_hidden_layers)

        #  HF PretrainedConfig 초기화
        # 여기서 kwargs는 config.json에 저장될 수 있고,
        # HF 생태계(AutoConfig/AutoModel)와 연결되는 기본 메커니즘이 동작한다.
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["SayaQwen2Config"]
