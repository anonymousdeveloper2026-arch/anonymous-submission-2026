from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional, Literal, Union
class SequenceLengthConfig(BaseModel):
    """Sequence length range configuration"""
    min: int = Field(..., ge=1, description="Minimum sequence length")
    max: int = Field(..., ge=1, description="Maximum sequence length")
    class Config:
        validate_assignment = True

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min must be <= max")
class AutoModelSelector(BaseModel):
    """HuggingFace model configuration"""
    name: str = Field(..., description="Model card name")
    pool_type: Literal["cls", "mean", "last", "pooler"] = Field(..., description="The pooling strategy to use ('cls', 'mean', or 'last'). This is now a required field.")
    prefix: Optional[str] = Field(default=None, description="Prefix to add to text before embedding (e.g., 'passage: ' for E5 models)")

    max_context_length: Optional[int] = Field(default=None, description="Maximum context length for the model (in tokenizer units). Used for qualification verification.")
class ByokEmbedderConfig(BaseModel):
    """BYOK embedder configuration"""
    provider: str = Field(..., description="API provider (e.g., 'openai', 'cohere', 'google')")
    name: str = Field(..., description="Litellm model name (e.g., 'text-embedding-3-large')")
    tokenizer_path: Optional[str] = Field(default=None, description="Tokenizer path for BYOK model (e.g., 'Cohere/cohere-tokenizer-fast'). If not specified, default tokenizer will be used which may result in lower accuracy.")

    max_context_length: Optional[int] = Field(default=None, description="Maximum context length for the model (in tokenizer units). Used for qualification verification.")
class LocalModelSelector(BaseModel):
    """Configuration for loading model classes directly from local Python files"""
    local_path: str = Field(..., description="Path to the .py file where the model class is defined")
    name: str = Field(..., description="Local model name")
    local_class: str = Field(..., description="Name of the class to load")
    pool_type: Literal["cls", "mean", "last", "pooler"] = Field(..., description="The pooling strategy to use ('cls', 'mean', or 'last'). This is now a required field.")
    max_context_length: Optional[int] = Field(default=None, description="Maximum context length for the model (in tokenizer units). Used for qualification verification.")
class ProbeConfig(BaseModel):
    """Probe generation configuration"""
    sequence_length: SequenceLengthConfig = Field(default=SequenceLengthConfig(min=4, max=16), description="Sequence length range. Evaluates sequentially from min to max.")
    samples_per_length: int = Field(default=10, description="Number of samples to evaluate for each sequence length.")
    token_per_sample: int = Field(default=256, description="Size of each chunk for that sequence length. Tokenizer is based on the embedder engine.")
    group_amount: int = Field(default=50, description="Only used for SRS score evaluation. Determines how many inverse-permutation and forward-permutation sets to collect for comparison.")
class ModelsConfig(BaseModel):
    merger: Optional[Union[AutoModelSelector, LocalModelSelector]] = Field(default=None, description="Model configuration for merger_mode (Hugging Face or local class)")
    base_embedder: Optional[Union[AutoModelSelector, LocalModelSelector]] = Field(default=None, description="Base embedder configuration (Hugging Face or local class)")
    native_embedder: Optional[Union[AutoModelSelector, LocalModelSelector]] = Field(default=None, description="Embedder configuration for native_mode (Hugging Face or local class)")
    byok_embedder: Optional[ByokEmbedderConfig] = Field(default=None, description="Embedder configuration for BYOK mode")
class DatasetConfig(BaseModel):
    """Dataset configuration"""
    path: str = Field(..., description="HuggingFace dataset path")
    split: str = Field(..., description="Dataset split (e.g., 'train', 'test')")
    text_column: str = Field(..., description="The name of the column containing the main text content.")
    commit_hash: str = Field(..., description="Revision/commit_hash of the HuggingFace dataset. Required for reproducibility.")
    data_dir: Optional[str] = Field(default=None, description="Specific data files to load (e.g., 'en'). Allows loading a subset of a large dataset.")
    data_files: Optional[str] = Field(default=None, description="Specific data files to load (e.g., '- en/c4-train.0000*-of-01024.json.gz'). Allows loading a subset of a large dataset. Overwrites the data_dir setting.")
class OutputConfig(BaseModel):
    format: str = Field(default="json")
    sign: bool = Field(default=True)
class BenchmarkConfig(BaseModel):
    metric: Literal["srs", "rss"] = Field(default="rss", description = "'rss' (default, Robustness to Sequence Scaling) or 'srs' (Sequence Recognition Sensitivity).")
    formula: Literal["q2q2", "q1q3"] = Field(default="q1q3", description = "q1q3 seperation score for default, q2q2 median comparison for optional selection")
    mode: Literal["merger_mode", "native_mode", "byok_mode"] = Field(default="merger_mode", description="merger_mode, native_mode, or byok_mode")
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    probe_config: ProbeConfig = Field(default_factory=ProbeConfig)
    datasets: list[DatasetConfig] = Field(default_factory=list, description="A list of dataset configurations to be used for sampling.")
    output: OutputConfig = Field(default_factory=OutputConfig)
    advanced: Dict[str, Any] = Field(default_factory=dict, description="Advanced options (batch_size, device, etc.)")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")

    @model_validator(mode='after')
    def validate_mode_config(self) -> 'BenchmarkConfig':
        if self.mode == "merger_mode":
            if self.models.merger is None:
                raise ValueError("merger_mode requires 'models.merger' configuration.")
            if self.models.base_embedder is None:
                raise ValueError("merger_mode requires 'models.base_embedder' configuration.")
        elif self.mode == "native_mode":
            if self.models.native_embedder is None:
                raise ValueError("native_mode requires 'models.native_embedder' configuration.")
        elif self.mode == "byok_mode":
            if self.models.byok_embedder is None:
                raise ValueError("byok_mode requires 'models.byok_embedder' configuration.")
        return self