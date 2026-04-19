import os
import sys
import json
import time
from typing import List
import mteb
from datetime import datetime
import torch
import numpy as np
import faiss
from typing import Any, List, Dict, Optional
from pathlib import Path
from tqdm import tqdm as tqdm_iter
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig
from mteb import SearchProtocol, TaskMetadata, get_model_meta
from mteb.models import ModelMeta
from mteb.types import RetrievalOutputType, CorpusDatasetType, EncodeKwargs, QueryDatasetType, TopRankedDocumentsType
import platform
import traceback

# split1/upper~ split4/lower
SPLIT = ""

TARGET_FOLDER = "examples/evals/lemb" + "/" + SPLIT

# please write the format like "Alibaba-NLP_gte-modernbert-base"
EXCLUDED_MODELS = [
    # "Alibaba-NLP_gte-base-en-v1.5",
]

# ================================================================== #
#  LongEmbed Task Instruction Utilities                                #
# ================================================================== #


def get_task_def_by_task_name_and_type(task_name: str, task_type: str) -> str:
    if task_type in ['STS']:
        return "Retrieve semantically similar text."

    if task_type in ['Summarization']:
        return "Given a news summary, retrieve other semantically similar summaries"

    if task_type in ['BitextMining']:
        return "Retrieve parallel sentences."

    if task_type in ['Classification']:
        task_name_to_instruct: Dict[str, str] = {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
            # C-MTEB eval instructions
            'TNews': 'Classify the fine-grained category of the given news title',
            'IFlyTek': 'Given an App description text, find the appropriate fine-grained category',
            'MultilingualSentiment': 'Classify sentiment of the customer review into positive, neutral, or negative',
            'JDReview': 'Classify the customer review for iPhone on e-commerce platform into positive or negative',
            'OnlineShopping': 'Classify the customer review for online shopping into positive or negative',
            'Waimai': 'Classify the customer review from a food takeaway platform into positive or negative',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Clustering']:
        task_name_to_instruct: Dict[str, str] = {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
            # C-MTEB eval instructions
            'CLSClusteringS2S': 'Identify the main category of scholar papers based on the titles',
            'CLSClusteringP2P': 'Identify the main category of scholar papers based on the titles and abstracts',
            'ThuNewsClusteringS2S': 'Identify the topic or theme of the given news articles based on the titles',
            'ThuNewsClusteringP2P': 'Identify the topic or theme of the given news articles based on the titles and contents',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Reranking', 'PairClassification']:
        task_name_to_instruct: Dict[str, str] = {
            'AskUbuntuDupQuestions': 'Retrieve duplicate questions from AskUbuntu forum',
            'MindSmallReranking': 'Retrieve relevant news articles based on user browsing history',
            'SciDocsRR': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            'StackOverflowDupQuestions': 'Retrieve duplicate questions from StackOverflow forum',
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
            # C-MTEB eval instructions
            'T2Reranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoReranking': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CMedQAv1': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'CMedQAv2': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'Ocnli': 'Retrieve semantically similar text.',
            'Cmnli': 'Retrieve semantically similar text.',
        }
        return task_name_to_instruct[task_name]

    if task_type in ['Retrieval']:
        if task_name.lower().startswith('cqadupstack'):
            return 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question'

        task_name_to_instruct: Dict[str, str] = {
            'ArguAna': 'Given a claim, find documents that refute the claim',
            'ClimateFEVER': 'Given a claim about climate change, retrieve documents that support or refute the claim',
            'DBPedia': 'Given a query, retrieve relevant entity descriptions from DBPedia',
            'FEVER': 'Given a claim, retrieve documents that support or refute the claim',
            'FiQA2018': 'Given a financial question, retrieve user replies that best answer the question',
            'HotpotQA': 'Given a multi-hop question, retrieve documents that can help answer the question',
            'MSMARCO': 'Given a web search query, retrieve relevant passages that answer the query',
            'MSMARCODOC': 'Given a web search query, retrieve relevant passages that answer the query',
            'Needles': 'Given a web search query, retrieve relevant passages that answer the query',
            'NFCorpus': 'Given a question, retrieve relevant documents that best answer the question',
            'NQ': 'Given a question, retrieve Wikipedia passages that answer the question',
            'QuoraRetrieval': 'Given a question, retrieve questions that are semantically equivalent to the given question',
            'SCIDOCS': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
            'SciFact': 'Given a scientific claim, retrieve documents that support or refute the claim',
            'Touche2020': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
            'TRECCOVID': 'Given a query on COVID-19, retrieve documents that answer the query',
            # C-MTEB eval instructions
            'T2Retrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'MMarcoRetrieval': 'Given a web search query, retrieve relevant passages that answer the query',
            'DuRetrieval': 'Given a Chinese search query, retrieve web passages that answer the question',
            'CovidRetrieval': 'Given a question on COVID-19, retrieve news articles that answer the question',
            'CmedqaRetrieval': 'Given a Chinese community medical question, retrieve replies that best answer the question',
            'EcomRetrieval': 'Given a user query from an e-commerce website, retrieve description sentences of relevant products',
            'MedicalRetrieval': 'Given a medical question, retrieve user replies that best answer the question',
            'VideoRetrieval': 'Given a video search query, retrieve the titles of relevant videos',
        }

        # add lower case keys to match some beir names
        task_name_to_instruct.update(
            {k.lower(): v for k, v in task_name_to_instruct.items()})
        # other cases where lower case match still doesn't work
        task_name_to_instruct['trec-covid'] = task_name_to_instruct['TRECCOVID']
        task_name_to_instruct['climate-fever'] = task_name_to_instruct['ClimateFEVER']
        task_name_to_instruct['dbpedia-entity'] = task_name_to_instruct['DBPedia']
        task_name_to_instruct['webis-touche2020'] = task_name_to_instruct['Touche2020']
        task_name_to_instruct['fiqa'] = task_name_to_instruct['FiQA2018']
        task_name_to_instruct['quora'] = task_name_to_instruct['QuoraRetrieval']

        # for miracl evaluation
        task_name_to_instruct['miracl'] = 'Given a question, retrieve Wikipedia passages that answer the question'

        return task_name_to_instruct[task_name]

    raise ValueError(
        f"No instruction config for task {task_name} with type {task_type}")


def get_detailed_instruct(task_description: str) -> str:
    if not task_description:
        return ''

    return 'Instruct: {}\nQuery: '.format(task_description)


# This was Fixed for the args.prefix_type == 'instruction'
INSTRUCT_PROMPT = get_detailed_instruct(
    get_task_def_by_task_name_and_type(task_name="Needles", task_type="Retrieval"))

# Device Checker

# Optional imports - gracefully handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def probe_system_info():
    """Probe system information."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        return {
            "mem": {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
            },
            "disk": {
                "total": disk.total,
                "available": disk.free,
            },
        }
    return {}


def probe_gpu_info():
    """Probe GPU information."""

    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot probe GPU info")
        return {}

    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = {
                "name": props.name,
                "total_memory": f"{props.total_memory / (1024**3):.2f} GB",
                "reserved_memory": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                "compute_cap": f"{props.major}.{props.minor}",
                "multi_procs": f"{props.multi_processor_count}",
            }
            gpus.append(gpu)

        return {
            "torch": {
                "version": str(torch.__version__),
                "avilable": torch.cuda.is_available(),
                "current_device": f"{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            },
            "cuda": {
                "version-cuda": str(torch.version.cuda),
                "version-cudnn": str(torch.backends.cudnn.version()),
                "gpus": gpus
            }
        }

    return {}


def probe_python_packages():
    """Probe installed Python packages."""
    packages_to_check = [
        ("torch", TORCH_AVAILABLE),
        ("transformers", TRANSFORMERS_AVAILABLE),
        ("accelerate", ACCELERATE_AVAILABLE),
        ("psutil", PSUTIL_AVAILABLE),
    ]

    # Add more common packages
    try:
        import huggingface_hub
        packages_to_check.append(("huggingface_hub", True))
    except ImportError:
        packages_to_check.append(("huggingface_hub", False))

    try:
        import sentence_transformers
        packages_to_check.append(("sentence_transformers", True))
    except ImportError:
        packages_to_check.append(("sentence_transformers", False))

    try:
        import numpy
        packages_to_check.append(("numpy", True))
    except ImportError:
        packages_to_check.append(("numpy", False))

    try:
        import pandas
        packages_to_check.append(("pandas", True))
    except ImportError:
        packages_to_check.append(("pandas", False))

    return packages_to_check


def probe_environment_variables():
    """Probe relevant environment variables."""

    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]

    res = {}
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        res[var] = value

    return res


def generate_device_info_dict():
    """Generate a dictionary containing device information for saving."""

    return {
        "sys_info": probe_system_info(),
        "gpu_info": probe_gpu_info(),
        "pkg_info": probe_python_packages(),
        "env_info": probe_environment_variables(),
    }


# ================================================================== #
#  ModelMTEBWrapper                                                    #
# ================================================================== #

class ModelMTEBWrapper(SearchProtocol):

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        max_ctx: int = 512,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        l2_norm: bool = True,
        pool_type: str = "mean",
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_ctx = max_ctx
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.l2_norm = l2_norm
        self.pool_type = pool_type
        self._load_encoder()
        self.encoder.eval()
        self._faiss_index: Optional[faiss.Index] = None
        self._corpus_ids: Optional[List[str]] = None
        self._mteb_model_meta: ModelMeta = get_model_meta(model_name)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        print(
            f"[ModelWrapper] Indexing corpus (split={hf_split}, subset={hf_subset})...")
        doc_ids, texts = self._extract_corpus_texts(corpus)
        self._corpus_ids = doc_ids
        embeddings = self._encode_texts(texts, is_query=False)
        dim = embeddings.shape[1]
        if self.l2_norm:
            self._faiss_index = faiss.IndexFlatIP(dim)
        else:
            self._faiss_index = faiss.IndexFlatL2(dim)
        self._faiss_index.add(embeddings.astype(np.float32))
        print(
            f"[ModelWrapper] Index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        if self._faiss_index is None or self._corpus_ids is None:
            raise RuntimeError("index() must be called before search().")
        print(
            f"[ModelWrapper] Searching queries (split={hf_split}, top_k={top_k})...")
        query_ids, query_texts = self._extract_query_texts(queries)
        query_embeddings = self._encode_texts(query_texts, is_query=True)
        scores, indices = self._faiss_index.search(
            query_embeddings.astype(np.float32), top_k)
        results: RetrievalOutputType = {}
        for q_idx, q_id in enumerate(query_ids):
            results[q_id] = {
                self._corpus_ids[doc_idx]: float(scores[q_idx, rank])
                for rank, doc_idx in enumerate(indices[q_idx])
                if doc_idx != -1
            }
        return results

    def _extract_corpus_texts(self, corpus) -> tuple[List[str], List[str]]:
        doc_ids, texts = [], []
        if isinstance(corpus, dict):
            for doc_id, doc in corpus.items():
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(str(doc_id))
                texts.append(combined)
        else:
            for row in corpus:
                doc_id = str(row.get("_id", row.get("id", "")))
                title = row.get("title", "")
                text = row.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(doc_id)
                texts.append(combined)
        return doc_ids, texts

    def _extract_query_texts(self, queries) -> tuple[List[str], List[str]]:
        query_ids, texts = [], []
        if isinstance(queries, dict):
            for q_id, q_text in queries.items():
                query_ids.append(str(q_id))
                texts.append(q_text if isinstance(q_text, str)
                             else q_text.get("text", ""))
        else:
            for row in queries:
                q_id = str(row.get("_id", row.get("id", "")))
                text = row.get("text", row.get("query", ""))
                query_ids.append(q_id)
                texts.append(text)
        return query_ids, texts

    def _encode_texts(self, texts: List[str], is_query: bool) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed_texts = [prefix + text for text in texts]
        embeddings = []
        with torch.no_grad():
            for i in tqdm_iter(range(0, len(prefixed_texts), self.batch_size),
                               desc="Encoding queries" if is_query else "Encoding corpus"):
                batch = prefixed_texts[i:i + self.batch_size]
                if self.pool_type == "last":
                    batch_dict = self.tokenizer(
                        batch, max_length=self.max_ctx - 1,
                        return_attention_mask=False, padding=False, truncation=True,
                    )
                    batch_dict['input_ids'] = [
                        ids + [self.tokenizer.eos_token_id] for ids in batch_dict['input_ids']
                    ]
                    inputs = self.tokenizer.pad(
                        batch_dict, padding=True, return_attention_mask=True, return_tensors='pt',
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=self.max_ctx, return_tensors="pt"
                    ).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.encoder(**inputs)
                    if self.pool_type == "mean":
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        emb = (outputs.last_hidden_state *
                               mask).sum(1) / mask.sum(1).clamp(1e-9)
                    elif self.pool_type == "cls":
                        emb = outputs.last_hidden_state[:, 0]
                    else:
                        seqlens = inputs["attention_mask"].sum(1) - 1
                        emb = outputs.last_hidden_state[range(
                            len(batch)), seqlens]
                    if self.l2_norm:
                        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                    embeddings.append(emb.cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)

    def _load_encoder(self):
        print(f"[ModelWrapper] Loading encoder: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.encoder = AutoModel.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.output_dim = self.encoder.config.hidden_size
        print(
            f"[ModelWrapper] Loaded {self.model_name} with dim={self.output_dim}, max_ctx={self.max_ctx}")

class SynthesizerMTEBWrapper(SearchProtocol):

    def __init__(
        self,
        synthesizer_model_name: str,
        e5_model_name: str = "intfloat/multilingual-e5-base",
        e5_pool_type: str = "mean",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        l2_norm: bool = True,
        prefix_type: str = "passage_only",
        max_input_tokens: int = 8190,
    ):
        self.e5_model_name = e5_model_name
        self.synthesizer_model_name = synthesizer_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.l2_norm = l2_norm
        self.prefix_type = prefix_type
        self.max_input_tokens = max_input_tokens

        self.e5_pool_type = e5_pool_type

        self.passage_prefix_len: int = 0
        self.query_prefix_len: int = 0

        self._load_e5_encoder()
        self._load_synthesizer()
        self.e5_encoder.eval()
        self.synthesizer.eval()


        self._faiss_index: Optional[faiss.Index] = None
        self._corpus_ids: Optional[List[str]] = None  

        self._mteb_model_meta: ModelMeta = get_model_meta(synthesizer_model_name)

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        print(f"[SynthesizerWrapper] Indexing corpus (split={hf_split}, subset={hf_subset})...")

        doc_ids, texts = self._extract_corpus_texts(corpus)
        self._corpus_ids = doc_ids

        embeddings = self._encode_texts(texts, is_query=False)  # [N, dim]

        dim = embeddings.shape[1]
        if self.l2_norm:
            self._faiss_index = faiss.IndexFlatIP(dim)   # Inner Product
        else:
            self._faiss_index = faiss.IndexFlatL2(dim)   # L2 distance

        self._faiss_index.add(embeddings.astype(np.float32))
        print(f"[SynthesizerWrapper] Index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        if self._faiss_index is None or self._corpus_ids is None:
            raise RuntimeError("index() must be called before search().")

        print(f"[SynthesizerWrapper] Searching queries (split={hf_split}, top_k={top_k})...")

        query_ids, query_texts = self._extract_query_texts(queries)
        query_embeddings = self._encode_texts(query_texts, is_query=True)  # [Q, dim]

        scores, indices = self._faiss_index.search(
            query_embeddings.astype(np.float32), top_k
        )  # [Q, top_k]

        results: RetrievalOutputType = {}
        for q_idx, q_id in enumerate(query_ids):
            results[q_id] = {
                self._corpus_ids[doc_idx]: float(scores[q_idx, rank])
                for rank, doc_idx in enumerate(indices[q_idx])
                if doc_idx != -1  
            }

        return results

    def _extract_corpus_texts(self, corpus) -> tuple[List[str], List[str]]:
        doc_ids, texts = [], []

        if isinstance(corpus, dict):
            for doc_id, doc in corpus.items():
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(str(doc_id))
                texts.append(combined)
        else:
            for row in corpus:
                doc_id = str(row.get("_id", row.get("id", "")))
                title = row.get("title", "")
                text = row.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(doc_id)
                texts.append(combined)

        return doc_ids, texts

    def _extract_query_texts(self, queries) -> tuple[List[str], List[str]]:
        query_ids, texts = [], []

        if isinstance(queries, dict):
            for q_id, q_text in queries.items():
                query_ids.append(str(q_id))
                texts.append(q_text if isinstance(q_text, str) else q_text.get("text", ""))
        else:
            for row in queries:
                q_id = str(row.get("_id", row.get("id", "")))
                text = row.get("text", row.get("query", ""))
                query_ids.append(q_id)
                texts.append(text)

        return query_ids, texts


    def _encode_texts(self, texts: List[str], is_query: bool) -> np.ndarray:
        embeddings = []
        for text in tqdm(texts, desc="Encoding queries" if is_query else "Encoding corpus"):
            chunks = self._chunk_text(text, is_query=is_query)
            chunk_embs = self._encode_chunks(chunks, is_query=is_query)
            final_emb = self._synthesize(chunk_embs)
            embeddings.append(final_emb)
        return np.stack(embeddings, axis=0)

    def _load_e5_encoder(self):
        print(f"[SynthesizerWrapper] Loading E5 encoder: {self.e5_model_name}")
        self.e5_tokenizer = AutoTokenizer.from_pretrained(self.e5_model_name, trust_remote_code=True)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.e5_encoder = AutoModel.from_pretrained(
            self.e5_model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.e5_dim = self.e5_encoder.config.hidden_size

        # Calculate actual prefix token lengths
        self.passage_prefix_len = len(self.e5_tokenizer.encode("passage: ", add_special_tokens=False))
        self.query_prefix_len = len(self.e5_tokenizer.encode("query: ", add_special_tokens=False))
        print(f"[SynthesizerWrapper] Prefix lengths - passage: {self.passage_prefix_len}, query: {self.query_prefix_len}")


    def _load_synthesizer(self):
        print(f"[SynthesizerWrapper] Loading : {self.synthesizer_model_name}")
        dtype = torch.float16 if self.use_fp16 else torch.float32
        config = AutoConfig.from_pretrained(self.synthesizer_model_name, trust_remote_code=True)
        self.synthesizer = AutoModel.from_pretrained(
            self.synthesizer_model_name, torch_dtype=dtype, config=config, trust_remote_code=True
        ).to(self.device)
        self.output_dim = getattr(config, "hidden_size", self.e5_dim)


    def _chunk_text(self, text: str, is_query: bool = False) -> List[str]:
        # Determine correct prefix length based on query/passage
        if self.prefix_type == "query_or_passage":
            prefix_len = self.query_prefix_len if is_query else self.passage_prefix_len
        elif self.prefix_type == "passage_only":
            prefix_len = self.passage_prefix_len
        else:
            prefix_len = 0
    
        # Calculate effective chunk size (accounting for prefix)
        effective_chunk_size = self.chunk_size - prefix_len
        if effective_chunk_size <= 0:
            raise ValueError(f"chunk_size ({self.chunk_size}) must be larger than prefix_len ({prefix_len})")
    
        # Truncate input text based on max_input_tokens
        total_tokens = self.e5_tokenizer.encode(text, add_special_tokens=False)
        if len(total_tokens) > self.max_input_tokens:
            total_tokens = total_tokens[:self.max_input_tokens]

        chunks = []
        start_idx = 0

        while start_idx < len(total_tokens):
            # Get chunk tokens using effective size
            end_idx = min(start_idx + effective_chunk_size, len(total_tokens))
            chunk_tokens = total_tokens[start_idx:end_idx]
    
            # Decode chunk back to text
            chunk_text = self.e5_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
    
            # Move start index forward by (effective_chunk_size - overlap)
            start_idx += effective_chunk_size - self.chunk_overlap
    
            # Prevent infinite loop if overlap >= effective_chunk_size
            if self.chunk_overlap >= effective_chunk_size:
                break

        return chunks

    def _encode_chunks(self, chunks: List[str], is_query: bool = False) -> np.ndarray:
        if self.prefix_type == "query_or_passage":
            prefix = "query: " if is_query else "passage: "
        if self.prefix_type == "passage_only":
            prefix = "passage: "
        chunks = [prefix + c for c in chunks]
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                inputs = self.e5_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.chunk_size, return_tensors="pt"
                ).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.e5_encoder(**inputs)
                    if self.e5_pool_type == "mean":
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)
                    elif self.e5_pool_type == "cls":
                        emb = outputs.last_hidden_state[:, 0]
                    else:
                        seqlens = inputs["attention_mask"].sum(1) - 1
                        emb = outputs.last_hidden_state[range(len(batch)), seqlens]
                embeddings.append(emb.cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)


    def _synthesize(self, chunk_embeddings: np.ndarray) -> np.ndarray:
        # Truncate to synthesizer's sequence limit (512 chunks max)
        if chunk_embeddings.shape[0] > 512:
            chunk_embeddings = chunk_embeddings[:512]
    
        chunk_tensor = torch.from_numpy(chunk_embeddings).to(self.device).unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                outputs = self.synthesizer(
                    inputs_embeds=chunk_tensor,
                    attention_mask=torch.ones(chunk_tensor.shape[:2], device=self.device)
                )
                final = outputs.pooler_output
                if self.l2_norm:
                    final = torch.nn.functional.normalize(final, p=2, dim=-1)
        return final.squeeze(0).cpu().float().numpy()

class AverageMTEBWrapper(SearchProtocol):

    def __init__(
        self,
        e5_model_name: str = "intfloat/multilingual-e5-base",
        pool_type: str = "mean",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        batch_size: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        l2_norm: bool = True,
        prefix_type: str = "passage_only",
        max_input_tokens: int = 8190,
    ):
        self.e5_model_name = e5_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.device = device
        self.use_fp16 = use_fp16
        self.l2_norm = l2_norm
        self.prefix_type = prefix_type
        self.max_input_tokens = max_input_tokens

        self.pool_type = pool_type

        self.passage_prefix_len: int = 0
        self.query_prefix_len: int = 0

        self._load_e5_encoder()
        self.e5_encoder.eval()

        self._faiss_index: Optional[faiss.Index] = None
        self._corpus_ids: Optional[List[str]] = None  

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return get_model_meta(self.e5_model_name)

    def index(
        self,
        corpus: CorpusDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        num_proc: int | None,
    ) -> None:
        print(f"[AverageWrapper] Indexing corpus (split={hf_split}, subset={hf_subset})...")

        doc_ids, texts = self._extract_corpus_texts(corpus)
        self._corpus_ids = doc_ids

        embeddings = self._encode_texts(texts, is_query=False)  # [N, dim]

        dim = embeddings.shape[1]
        if self.l2_norm:
            self._faiss_index = faiss.IndexFlatIP(dim)   # Inner Product
        else:
            self._faiss_index = faiss.IndexFlatL2(dim)   # L2 distance

        self._faiss_index.add(embeddings.astype(np.float32))
        print(f"[AverageWrapper] Index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def search(
        self,
        queries: QueryDatasetType,
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        top_k: int,
        encode_kwargs: EncodeKwargs,
        top_ranked: TopRankedDocumentsType | None = None,
        num_proc: int | None,
    ) -> RetrievalOutputType:
        if self._faiss_index is None or self._corpus_ids is None:
            raise RuntimeError("index() must be called before search().")

        print(f"[AverageWrapper] Searching queries (split={hf_split}, top_k={top_k})...")

        query_ids, query_texts = self._extract_query_texts(queries)
        query_embeddings = self._encode_texts(query_texts, is_query=True)  # [Q, dim]

        scores, indices = self._faiss_index.search(
            query_embeddings.astype(np.float32), top_k
        )  # [Q, top_k]

        results: RetrievalOutputType = {}
        for q_idx, q_id in enumerate(query_ids):
            results[q_id] = {
                self._corpus_ids[doc_idx]: float(scores[q_idx, rank])
                for rank, doc_idx in enumerate(indices[q_idx])
                if doc_idx != -1 
            }

        return results

    def _extract_corpus_texts(self, corpus) -> tuple[List[str], List[str]]:
        doc_ids, texts = [], []

        if isinstance(corpus, dict):
            for doc_id, doc in corpus.items():
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(str(doc_id))
                texts.append(combined)
        else:
            for row in corpus:
                doc_id = str(row.get("_id", row.get("id", "")))
                title = row.get("title", "")
                text = row.get("text", "")
                combined = f"{title} {text}".strip() if title else text
                doc_ids.append(doc_id)
                texts.append(combined)

        return doc_ids, texts

    def _extract_query_texts(self, queries) -> tuple[List[str], List[str]]:
        query_ids, texts = [], []

        if isinstance(queries, dict):
            for q_id, q_text in queries.items():
                query_ids.append(str(q_id))
                texts.append(q_text if isinstance(q_text, str) else q_text.get("text", ""))
        else:
            for row in queries:
                q_id = str(row.get("_id", row.get("id", "")))
                text = row.get("text", row.get("query", ""))
                query_ids.append(q_id)
                texts.append(text)

        return query_ids, texts

    def _encode_texts(self, texts: List[str], is_query: bool) -> np.ndarray:
        embeddings = []
        for text in tqdm(texts, desc="Encoding queries" if is_query else "Encoding corpus"):
            chunks = self._chunk_text(text, is_query=is_query)
            chunk_embs = self._encode_chunks(chunks, is_query=is_query)
            final_emb = self._synthesize_with_average(chunk_embs)
            embeddings.append(final_emb)
        return np.stack(embeddings, axis=0)


    def _load_e5_encoder(self):
        print(f"[AverageWrapper] Loading E5 encoder: {self.e5_model_name}")
        self.e5_tokenizer = AutoTokenizer.from_pretrained(self.e5_model_name, trust_remote_code=True)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.e5_encoder = AutoModel.from_pretrained(
            self.e5_model_name, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.e5_dim = self.e5_encoder.config.hidden_size

        # Calculate actual prefix token lengths
        self.passage_prefix_len = len(self.e5_tokenizer.encode("passage: ", add_special_tokens=False))
        self.query_prefix_len = len(self.e5_tokenizer.encode("query: ", add_special_tokens=False))
        print(f"[AverageWrapper] Prefix lengths - passage: {self.passage_prefix_len}, query: {self.query_prefix_len}")
    def _chunk_text(self, text: str, is_query: bool = False) -> List[str]:
        """
        Naive fixed-length chunking using only e5_tokenizer.
        Splits text into chunks of effective_chunk_size tokens with overlap.
        Accounts for prefix length to prevent data truncation.
        """
        # Determine correct prefix length based on query/passage
        if self.prefix_type == "query_or_passage":
            prefix_len = self.query_prefix_len if is_query else self.passage_prefix_len
        elif self.prefix_type == "passage_only":
            prefix_len = self.passage_prefix_len
        else:
            prefix_len = 0
    
        # Calculate effective chunk size (accounting for prefix)
        effective_chunk_size = self.chunk_size - prefix_len
        if effective_chunk_size <= 0:
            raise ValueError(f"chunk_size ({self.chunk_size}) must be larger than prefix_len ({prefix_len})")
    
        # Truncate input text based on max_input_tokens
        total_tokens = self.e5_tokenizer.encode(text, add_special_tokens=False)
        if len(total_tokens) > self.max_input_tokens:
            total_tokens = total_tokens[:self.max_input_tokens]

        chunks = []
        start_idx = 0

        while start_idx < len(total_tokens):
            # Get chunk tokens using effective size
            end_idx = min(start_idx + effective_chunk_size, len(total_tokens))
            chunk_tokens = total_tokens[start_idx:end_idx]
    
            # Decode chunk back to text
            chunk_text = self.e5_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
    
            # Move start index forward by (effective_chunk_size - overlap)
            start_idx += effective_chunk_size - self.chunk_overlap
    
            # Prevent infinite loop if overlap >= effective_chunk_size
            if self.chunk_overlap >= effective_chunk_size:
                break

        return chunks

    def _encode_chunks(self, chunks: List[str], is_query: bool = False) -> np.ndarray:
        if self.prefix_type == "query_or_passage":
            prefix = "query: " if is_query else "passage: "
        if self.prefix_type == "passage_only":
            prefix = "passage: "
        chunks = [prefix + c for c in chunks]
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                inputs = self.e5_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.chunk_size, return_tensors="pt"
                ).to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    outputs = self.e5_encoder(**inputs)
                    if self.pool_type == "mean":
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)
                    elif self.pool_type == "cls":
                        emb = outputs.last_hidden_state[:, 0]
                    else:
                        seqlens = inputs["attention_mask"].sum(1) - 1
                        emb = outputs.last_hidden_state[range(len(batch)), seqlens]
                embeddings.append(emb.cpu().float().numpy())
        return np.concatenate(embeddings, axis=0)

    def _synthesize_with_average(self, chunk_embeddings: np.ndarray) -> np.ndarray:
        # Simple mean pooling across all chunks
        final_emb = np.mean(chunk_embeddings, axis=0)

        if self.l2_norm:
            # L2 normalize the final vector
            norm = np.linalg.norm(final_emb)
            if norm > 0:
                final_emb = final_emb / norm

        return final_emb


# ================================================================== #
#  Automated Batch Evaluation                                          #
# ================================================================== #


# LongEmbed evaluation protocol constants
TASK_LIST = [
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBSummScreenFDRetrieval",
    "LEMBQMSumRetrieval",
    "LEMBWikimQARetrieval",
    "LEMBNarrativeQARetrieval"
]
# WINDOW_LENGTH_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# We stripped the model's max_ctx to 8190. See the main paper and appendix for the detailed reason for this setting.
# This is the only difference from the original LEMB scoring.
WINDOW_LENGTH_LIST = [256, 512, 1024, 2048, 4096, 8192]

NEEDLE_PASSKEY_TASKS = ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]
RETRIEVAL_TASKS = ["LEMBSummScreenFDRetrieval", "LEMBQMSumRetrieval",
                   "LEMBWikimQARetrieval", "LEMBNarrativeQARetrieval"]

def evaluate_single_model(config_path: str, output_dir: str):
    """Load config.json, build model, run full LongEmbed evaluation, save results."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_name = config['model_name']

    # Dynamically select the appropriate wrapper based on config
    # Mode selection: "E5-SYNTH", "E5-AVERAGE", or "NATIVE-ENCODER"
    detected_mode:str = config.get('mode', 'NATIVE-ENCODER')
    detected_mode = detected_mode.upper()
    if detected_mode == 'NATIVE-ENCODER' or detected_mode == 'E5-SYNTH' or detected_mode == 'E5-AVERAGE':
        pass
    else:
        raise Exception("Mode config must be one of : 'E5-SYNTH', 'E5-AVERAGE', or 'NATIVE-ENCODER'")

    if detected_mode == 'E5-SYNTH':
        # Synthesizer mode: use E5 + Synthesizer pipeline
        print(f"  [LEMB] Detected Synth configuration: {config['model_name']}")
        model = SynthesizerMTEBWrapper(
            synthesizer_model_name=config['model_name'],
            e5_model_name=config.get('e5_model_name', 'intfloat/multilingual-e5-base'),
        )
    elif detected_mode == 'E5-AVERAGE':
        # Average mode: use E5 + Average pooling pipeline
        print(f"  [LEMB] Detected Average configuration: ignore model name")
        model = AverageMTEBWrapper(
            e5_model_name=config.get('e5_model_name', 'intfloat/multilingual-e5-base'),
        )     
    else:
        # Standard mode: use single encoder
        model_name = config['model_name']
        query_prefix = config['query_prefix']
        passage_prefix = config['passage_prefix']
        max_ctx = config['max_ctx']
        pool_type = config['pool_type']
        batch_size = config['batch_size']
        # If query_prefix is empty, follow LongEmbed protocol with INSTRUCT_PROMPT
        if not query_prefix:
            query_prefix = INSTRUCT_PROMPT

        print(f"  [LEMB] Detected Native configuration: {config['model_name']}")
        model = ModelMTEBWrapper(
            model_name=model_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            max_ctx=max_ctx,
            pool_type=pool_type,
            batch_size=batch_size,
        )

    output_dict = {}

    # --- Needle & Passkey tasks ---
    needle_passkey_task_list = [
        t for t in TASK_LIST if t in NEEDLE_PASSKEY_TASKS]
    if needle_passkey_task_list:
        print(
            f"  [LEMB] Evaluating needle/passkey tasks: {needle_passkey_task_list}")
        context_length_list = sorted(WINDOW_LENGTH_LIST)
        results = mteb.evaluate(
            model=model,
            prediction_folder=output_dir,
            tasks=mteb.get_tasks(needle_passkey_task_list),
        )
        eval_dict_list = results.model_dump()["task_results"]
        for task_result in eval_dict_list:
            task_name = task_result["task_name"]
            scores = task_result["scores"]
            needle_passkey_score_list = []
            for ctx_len in context_length_list:
                test_key = f"test_{ctx_len}"
                if test_key in scores:
                    for score_dict in scores[test_key]:
                        if "ndcg_at_1" in score_dict:
                            needle_passkey_score_list.append(
                                [ctx_len, score_dict["ndcg_at_1"]])
                            break
            if needle_passkey_score_list:
                avg_score = sum(
                    [x[1] for x in needle_passkey_score_list]) / len(needle_passkey_score_list)
                needle_passkey_score_list.append(["avg", avg_score])
                output_dict[task_name] = {item[0]: item[1]
                                          for item in needle_passkey_score_list}

    # --- Retrieval tasks ---
    retrieval_task_list = [t for t in TASK_LIST if t in RETRIEVAL_TASKS]
    if retrieval_task_list:
        print(f"  [LEMB] Evaluating retrieval tasks: {retrieval_task_list}")
        results = mteb.evaluate(
            model=model,
            prediction_folder=output_dir,
            tasks=mteb.get_tasks(retrieval_task_list),
        )
        eval_dict_list = results.model_dump()["task_results"]
        for task_result in eval_dict_list:
            task_name = task_result["task_name"]
            scores = task_result["scores"]
            split = "test" if "test" in scores else "validation"
            ndcg_at_1 = None
            ndcg_at_10 = None
            for score_dict in scores[split]:
                if "ndcg_at_1" in score_dict:
                    ndcg_at_1 = score_dict["ndcg_at_1"]
                if "ndcg_at_10" in score_dict:
                    ndcg_at_10 = score_dict["ndcg_at_10"]
                if ndcg_at_1 is not None and ndcg_at_10 is not None:
                    break
            output_dict[task_name] = {
                "ndcg@1": ndcg_at_1, "ndcg@10": ndcg_at_10}

    # --- Save results ---
    print("  " + "=" * 50)
    print(f"  [LEMB] RESULTS for {model_name}")
    print("  " + "=" * 50)
    print(json.dumps(output_dict, indent=2))

    results_file = os.path.join(output_dir, 'overall_results.json')
    with open(results_file, 'w') as f:
        json.dump(output_dict, f, indent=4)
    print(f"  [LEMB] Results saved to: {results_file}")

    # Cleanup GPU memory
    del model
    torch.cuda.empty_cache()


def main(target_folder):
    base_path = Path(target_folder)
    status_log_path = base_path / "status.log"
    print(f"🔥 Starting stateful LEMB evaluation across: {base_path.resolve()}")
    print(f"   Status Log: {status_log_path}")

    # 1. Read status.log to find completed models
    completed_models = set()
    if status_log_path.exists():
        with open(status_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    completed_models.add(parts[1])

    print(
        f"Found {len(completed_models)} previously processed models in status log.")

    # 2. Find all model configs and filter out completed ones
    all_config_files = sorted(base_path.rglob('config.json'))
    models_to_evaluate = []
    for config_path in all_config_files:
        model_name = config_path.parent.name
        if model_name not in completed_models and model_name not in EXCLUDED_MODELS:
            models_to_evaluate.append(config_path)

    if not models_to_evaluate:
        print("✨ All models have already been evaluated. Nothing to do. ✨")
        return

    print(f"Found {len(models_to_evaluate)} new models to evaluate.")

    # 3. Loop through the remaining models
    progress_bar = tqdm(models_to_evaluate, desc="Evaluation Progress")

    for config_path in progress_bar:
        model_name = config_path.parent.name
        progress_bar.set_description(f"🚀 Running LEMB for [{model_name}]")
        output_dir = str(config_path.parent)
        status = "FAIL"  # Default status

        try:
            start_time = time.monotonic()
            evaluate_single_model(str(config_path), output_dir)
            end_time = time.monotonic()

            # Write timing and device info to report.txt
            report_path = os.path.join(output_dir, "report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                report_data = {
                    "evaluation": {"total_wall_clock_elapsed": end_time - start_time},
                    "env": generate_device_info_dict()
                }
                f.write(json.dumps(report_data, indent=4))

            status = "SUCCESS"
            print(f"  ✅ FINISHED: LEMB for [{model_name}]")

        except Exception as e:
            error_log_path = os.path.join(output_dir, 'error.log')
            error_message = f"---\nERROR during LEMB for [{model_name}]\nConfig: {config_path}\nError: {e}\n---"
            print(f"  🚨 FAILED: LEMB for [{model_name}]. See {error_log_path}")
            with open(error_log_path, "w", encoding='utf-8') as f:
                f.write(error_message)
                f.write('\n')
                f.write(traceback.format_exception())

        finally:
            # 4. Log status regardless of outcome
            with open(status_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{model_name},{status}\n")
            if status == 'FAIL':
                break

    print("\n\n⏸️ LEMB evaluations loop halted! ⏸️")


if __name__ == '__main__':
    main(TARGET_FOLDER)
