import argparse
import logging
from typing import List, Dict, Union, Tuple, Optional

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from torch import Tensor
from tqdm import trange
from transformers import AutoModel, AutoTokenizer, AutoConfig

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


class SentenceGPTHF:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", sharded: bool = False,
                 device: Optional[str] = None, query_batch_size_multiplier: int = 1, **kwargs):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(self.device))
        else:
            self.device = "cpu"
        self.sep = sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if sharded:
            config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = AutoModel.from_config(config)
            model.tie_weights()
            self.model = load_checkpoint_and_dispatch(
                model, model_path, device_map="auto", no_split_module_classes=["GPTNeoBlock"]
            )
            logging.info(f"Device Map: {self.model.hf_device_map.items()}")
        else:
            model = AutoModel.from_pretrained(model_path)
            self.model = model.to(self.device)
        self.model.eval()
        self._sharded = sharded
        self._query_batch_size_multiplier = query_batch_size_multiplier
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True, max_length=self.tokenizer.model_max_length - 2)
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt").to(self.device)
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens) -> Tensor:
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True,
                                           return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def _batch_processing(self, texts: List[str], batch_size: int, is_query: bool, show_progress_bar: bool = True) -> Tensor:
        all_embeddings = []
        for start_index in trange(0, len(texts), batch_size, desc="Batches", disable=not show_progress_bar):
            texts_batch = texts[start_index:start_index+batch_size]
            embeddings = self.get_weightedmean_embedding(self.tokenize_with_specb(texts_batch, is_query=is_query))
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings)
        return all_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 128, show_progress_bar: bool = True, **kwargs) -> Union[
        List[Tensor], np.ndarray, Tensor]:
        length_sorted_idx: np.ndarray = np.argsort([-len(query) for query in queries])
        queries_sorted = [queries[idx] for idx in length_sorted_idx]
        all_embeddings = self._batch_processing(queries_sorted, batch_size * self._query_batch_size_multiplier,
                                                is_query=True, show_progress_bar=show_progress_bar)
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]
        return all_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 128,  show_progress_bar: bool = True, **kwargs) -> Union[
        List[Tensor], np.ndarray, Tensor]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        all_embeddings = self._batch_processing(sentences, batch_size, is_query=False, show_progress_bar=show_progress_bar)
        return all_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", required=True, help="path to corpus.jsonl")
    parser.add_argument("--query_file", required=True, help="path to queries.jsonl")
    parser.add_argument("--qrels_file", required=True, help="path to qrels.tsv")
    parser.add_argument("--model_path", required=True,
                        help="huggingface model repository name or local path (for sharded it must be a local path)")
    parser.add_argument("--sharded", action="store_ture")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--query_batch_size_multiplier", default=8, type=int)
    parser.add_argument("--score_function", default="cos_sim", type=str, choices=["cos_sim", "dot"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=args.corpus_file,
        query_file=args.query_file,
        qrels_file=args.qrels_file
    ).load_custom()

    model = DRES(
        SentenceGPTHF(args.model_path, sharded=True, query_batch_size_multiplier=args.query_batch_size_multiplier),
        batch_size=args.batch_size
    )

    retriever = EvaluateRetrieval(model, score_function=args.score_function)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
