from typing import Optional
import torch
from transformers import AutoModel, AutoTokenizer
# import litellm 
import tiktoken
import warnings

from .interfaces import FinesseEmbedder, FinesseSynthesizer

class HuggingFaceEmbedder(FinesseEmbedder):
    """
    Hugging Face 모델을 사용하는 임베딩 엔진 구현체.
    """
    
    def __init__(self, model_path: str, pool_type: str, prefix: Optional[str] = None, device: Optional[str] = None, dtype: Optional[torch.dtype] = None, max_length: int = 512):
        """
        Args:
            model_path: Hugging Face 모델 경로
            pool_type: 풀링 방식 ('mean', 'cls', 'last')
            prefix: 임베딩 전 텍스트에 추가할 접두사 (예: "passage: " for E5)
            device: Optional device to use (e.g., 'cuda', 'cpu'). Defaults to auto-detect.
            dtype: Optional data type for the model (e.g., torch.float16). Defaults to auto-detect based on device.
            max_length: Maximum token length for embedding (default: 512)
        """
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == 'cuda' else torch.float32)
        self.model_path = model_path
        self.pool_type = pool_type
        self.prefix = prefix or ""
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=self.dtype
        ).to(self.device).eval()

    def encode(self, texts: list[str]) -> torch.Tensor:
        """텍스트 리스트를 임베딩하여 반환한다."""
        # Add prefix if specified
        input_texts = [self.prefix + text for text in texts]

        with torch.no_grad():
            if self.pool_type == "last":
                # Tokenize for last-token pooling, ensuring EOS token is always present
                batch_dict = self.tokenizer(
                    input_texts,
                    max_length=self.max_length - 1,
                    return_attention_mask=False,
                    padding=False,
                    truncation=True,
                )
                batch_dict['input_ids'] = [
                    ids + [self.tokenizer.eos_token_id] for ids in batch_dict['input_ids']
                ]
                inputs = self.tokenizer.pad(
                    batch_dict,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                ).to(self.device)

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                mask = inputs["attention_mask"]
                seq_lengths = mask.sum(dim=1) - 1
                embeddings = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]

            else: # For 'cls' and 'mean' pooling
                inputs = self.tokenizer(
                    input_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
            
                if self.pool_type == "cls":
                    embeddings = hidden_states[:, 0]
                elif self.pool_type == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    embeddings = (hidden_states * mask).sum(1) / mask.sum(1).clamp(1e-9)
                else:
                    raise ValueError(f"Unknown or unsupported pool_type for this path: {self.pool_type}")

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    
    def device(self) -> torch.device:
        return self.device

    def count_tokens(self, text: str) -> int:
        """주어진 텍스트의 토큰 수를 반환한다."""
        # Use the embedder's own tokenizer to count tokens
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str, max_tokens: int) -> str:
        """주어진 텍스트를 최대 토큰 수만큼 정확히 잘라내서 반환한다."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Take only the first max_tokens tokens
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        
        # Decode back to text
        return self.tokenizer.decode(tokens)

class HuggingFaceSynthesizer(FinesseSynthesizer):
    def __init__(self, model_path: str, pool_type: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.type == 'cuda' else torch.float32)
        self.model_path = model_path
        self.pool_type = pool_type
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=self.dtype
        ).to(self.device).eval()
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Ensure input is on correct device and dtype
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
    
        # Run through model
        with torch.no_grad():
            outputs = self.model(embeddings)
        
            # Get hidden states for pooling
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'pooler_output'):
                # If pooler_output exists but we want custom pooling, we need hidden states
                # Try to get from outputs[0] if it's a tuple
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    # Fallback: use pooler_output if we can't get hidden states
                    hidden_states = None
            elif isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
        
            # Apply configured pooling
            if hidden_states is not None:
                if self.pool_type == "cls":
                    synth_emb = hidden_states[:, 0]
                elif self.pool_type == "mean":
                    synth_emb = hidden_states.mean(dim=1)
                elif self.pool_type == "last":
                    synth_emb = hidden_states[:, -1]
                elif self.pool_type == "pooler":
                    synth_emb = outputs.pooler_output        
                else:
                    raise ValueError(f"Unknown pool_type: {self.pool_type}")
            else:
                # Fallback to pooler_output if available
                synth_emb = outputs.pooler_output        
        # Normalize
        synth_emb = torch.nn.functional.normalize(synth_emb, p=2, dim=1)
        return synth_emb
    
    
    def device(self) -> torch.device:
        return self.device


class NullSynthesizer(FinesseSynthesizer):
    def __init__(self):
        self.device = torch.device("cpu")  # CPU 연산
    
    def synthesize(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Mean pooling
        synth_emb = embeddings.mean(dim=1)
        
        # Normalize
        synth_emb = torch.nn.functional.normalize(synth_emb, p=2, dim=1)
        return synth_emb
    
    
    def device(self) -> torch.device:
        return self.device