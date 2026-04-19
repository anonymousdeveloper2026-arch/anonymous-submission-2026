from abc import ABC, abstractmethod
from typing import Optional
import torch


class FinesseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def chunk_text(self, text: str, max_tokens: int) -> str:
        pass


class FinesseSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def device(self) -> torch.device:
        pass