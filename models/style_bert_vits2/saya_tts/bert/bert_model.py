import torch
from transformers import AutoTokenizer, AutoModel


class StyleBertModel:
    """
    Style-Bert-VITS2의 BERT 경로를 재현한 inference 전용 래퍼.

    핵심 설계 원칙:
    - BERT는 '문맥 특징 추출기'다
    - pooler / cls / MLM head는 쓰지 않는다
    - last_hidden_state만 TextEncoder로 보낸다
    """
    def __init__(self, model_name: str, device: str):
        self.device = device

        # Tokenizer
        # 원본 repo는 일본어 char-level WWM 모델을 전제로 한다.
        # add_prefix_space=True 는 DeBERTa 계열에서 중요
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            add_prefix_space=True,
        )

        # Model
        # AutoModel -> encoder 본체만 로드 (pooler 없음)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # hidden size
        # TextEncoder의 bert_proj Conv1d 입력 채널 수와 맞아야 한다
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def forward(self, text: str) -> torch.Tensor:
        """
        Args:
            text (str): 일본어 입력 문장

        Returns:
            torch.Tensor:
              shape = (1, T_bert, hidden_size)
              dtype = float32

        이 텐서는 그대로 TextEncoder의 bert_proj에 들어간다.
        """
        # tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # forward
        outputs = self.model(**inputs)

        # Style-Bert-VITS2는 last_hidden_state만 사용
        hidden_states = outputs.last_hidden_state

        # 안전하게 float32로 통일
        return hidden_states.float()
