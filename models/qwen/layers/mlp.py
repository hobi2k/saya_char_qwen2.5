from torch import nn
import torch

from transformers.activations import ACT2FN

class Qwne2MLP(nn.Module):
    """
    Qwen2 Feed-Forward Network (FFN) / MLP block.

    Multi-Layer Perceptron (MLP)

    기본 구조: Linear -> Activation -> Linear

    Attention이 토큰들끼리 서로 참고하는 단계라면,
    MLP는 각 토큰이 독립적으로 연산을 수행하는 단계다.

    이 클래스는 Transformer Decoder Block 내부의
    "Position-wise Feed Forward Network"에 해당한다.

    Qwen2 / LLaMA 계열은 일반적인 FFN(Dense -> Act -> Dense)이 아니라
    Gated MLP (SwiGLU 계열)를 사용한다.

    수식 관점에서 이 MLP는 다음을 수행한다:

        FFN(x) = W_down( act(W_gate(x)) ⊙ W_up(x) )

    여기서:
        - ⊙ : element-wise multiplication
        - act : SiLU / GELU 등 비선형 함수
    """
    def __init__(self, config):
        super().__init__()

        # 1. 핵심 차원 정리

        # hidden_size
        # - Transformer 내부 토큰 표현 차원
        # - Qwen2-7B: 4096
        self.hidden_size = config.hidden_size

        # intermediate_size
        # - FFN 내부 확장 차원
        # - hidden_size보다 훨씬 큼 (보통 4~8배)
        # - Qwen2-7B -> 22016
        self.intermediate_size = config.intermediate_size

        # 2. Gated MLP를 구성하는 Linear 계층
        
        # gate_proj:
        # - 입력 x를 gate용 벡터로 변환
        # - 이후 activation(SiLU 등)을 통과함
        #
        # shape:
        # input : (batch, seq_len, hidden_size)
        # output: (batch, seq_len, intermediate_size)
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,  # Qwen2 / LLaMA 계열은 bias 사용 안 함
        )


        # up_proj:
        # - 입력 x를 "value branch"로 변환
        # - activation 없이 gate 결과와 곱해짐

        # shape:
        # input : (batch, seq_len, hidden_size)
        # output: (batch, seq_len, intermediate_size)
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
        )

        # down_proj:
        # - gated 결과를 다시 hidden_size로 축소

        # shape:
        # input : (batch, seq_len, intermediate_size)
        # output: (batch, seq_len, hidden_size)
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
        )

        # 3. Activation Function

        # config.hidden_act 예:
        # - "silu"
        # - "gelu"
        #
        # ACT2FN은 문자열 -> 함수 매핑 dict
        # Hugging Face Transformers 라이브러리 내부에 정의되어 있는 유틸리티
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Qwen2 MLP

        Args:
            x (torch.Tensor):
            - 입력 hidden states
            - shape = (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor:
            - FFN 출력
            - shape = (batch_size, seq_len, hidden_size)
        """
        # 1. Gate branch

        # gate_proj(x):
        # - (B, S, H) -> (B, S, I)
        #
        # act(...):
        # - 비선형성 부여
        gate = self.act_fn(self.gate_proj(x))

        # 2. Up branch

        # up_proj(x):
        # - (B, S, H) -> (B, S, I)
        up = self.up_proj(x)

        # 3. Element-wise gating

        # gated activation:
        # - gate ⊙ up
        #
        # 이 구조가 중요한 이유:
        # - 단순 FFN보다 표현력이 훨씬 강함
        # - LLaMA, Qwen, Mistral 모두 채택
        gated = gate * up

        # 4. Down projection

        # (B, S, I) -> (B, S, H)
        out = self.down_proj(gated)

        return out