from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SayaTTSConfig:
    """
    Inference 전용 설정.
    학습 관련 필드는 의도적으로 제거한다.
    """

    # checkpoint / config
    ckpt_path: Path
    hps_path: Path

    # BERT
    # Style-Bert-VITS2 원본에서 쓰는 모델을 그대로 고정
    bert_model_name: str = (
        "ku-nlp/deberta-v2-large-japanese-char-wwm"
    )

    # audio
    sampling_rate: int = 44100

    # runtime
    device: str = "cuda"
