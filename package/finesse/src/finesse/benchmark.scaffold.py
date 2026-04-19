import torch
from typing import List
# Optional: for loading configs
from transformers import AutoConfig, PreTrainedModel

from finesse.interfaces import FinesseEmbedder, FinesseSynthesizer

# --- Custom Embedder Example ---
# Uncomment and customize this class for your embedder.


class MyCustomEmbedder(FinesseEmbedder):
    """
    Custom embedder model.
    Replace this docstring with a description of your model.
    """

    def __init__(self, config_path: str):
        """
        Initialize the model.
        config_path points to the directory containing your config.json or similar.
        Load your model weights, tokenizer, etc. here.
        """
        super().__init__()

        # Example: Load config (optional)
        # self.config = AutoConfig.from_pretrained(config_path)

        # TODO: Load your model here, e.g.:
        # self.model = YourModelClass.from_pretrained(config_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(config_path)

        print(f"{{self.__class__.__name__}} initialized from {{config_path}}")

    def embed(self, texts: List[str], **kwargs) -> torch.Tensor:
        """
        Generate embeddings for the input texts.

        Args:
            texts: List of input strings to embed.
            **kwargs: Additional keyword arguments (e.g., batch_size).

        Returns:
            torch.Tensor of shape (len(texts), embedding_dim).
        """
        # TODO: Implement your embedding logic here.
        # Example:
        # inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        # embeddings = outputs.last_hidden_state.mean(dim=1)  # Or use pooler_output
        # return embeddings

        raise NotImplementedError(
            "Implement the embed method for your custom model.")

    def device(self):
        return "cpu"
# --- Custom Synthesizer Example ---
# Uncomment and customize this for a synthesizer (e.g., sequence merger).
#
# class MyCustomSynthesizer(FinesseSynthesizer):
#     def __init__(self, config_path: str):
#         super().__init__()
#         # TODO: Load synthesizer model
#         pass
#
#     def synthesize(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
#         # TODO: Implement synthesis logic (e.g., merge sequence embeddings)
#         raise NotImplementedError("Implement the synthesize method.")
#     def device(self):
#         return "cpu"
