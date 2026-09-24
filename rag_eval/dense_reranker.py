"""Score dense candidates with xiaojev, then preserve both ranking signals."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_eval.common import PROP_RELEVANCE, passage_block
from rag_eval.fusion import fuse_rankings


class DenseReranker:
    def __init__(self, checkpoint, config_path):
        import torch
        from transformers import AutoTokenizer

        from training.train import MODEL_PATH, StudentModel, load_ckpt

        self.torch = torch
        self.config = json.loads(Path(config_path).read_text())["config"]
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = StudentModel().to("cuda")
        load_ckpt(self.model, checkpoint)
        self.model.eval()

    def rerank(self, question, passages):
        """passages: dense-ordered dictionaries containing docid, title, text."""
        from training.train import collate, encode_row, microbatch

        docids = [p["docid"] for p in passages]
        if len(set(docids)) != len(docids):
            raise ValueError("Duplicate retrieval candidates")
        paths, slices = [], []
        for passage in passages:
            row = {
                "primitive": "noul",
                "candidates": ["yes", "no"],
                "proposition": PROP_RELEVANCE,
                "state": f"Question: {question}\n\n{passage_block(passage['title'], passage['text'])}",
            }
            _, _, candidate_paths = encode_row(self.tokenizer, row)
            slices.append((len(paths), len(candidate_paths)))
            paths.extend(candidate_paths)
        scores = {}
        torch = self.torch
        with torch.inference_mode():
            for indices in microbatch(slices, paths, 16384):
                batch_paths = [
                    paths[j]
                    for i in indices
                    for j in range(slices[i][0], sum(slices[i]))
                ]
                tokens, mask, lengths = collate(
                    batch_paths, self.tokenizer.pad_token_id, "cuda"
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(tokens, mask, lengths)
                for offset, i in enumerate(indices):
                    scores[docids[i]] = round(
                        float(
                            torch.softmax(logits[2 * offset : 2 * offset + 2], dim=0)[0]
                        ),
                        6,
                    )
        ordered = fuse_rankings(docids, scores, **self.config)
        by_id = {p["docid"]: p for p in passages}
        return [{**by_id[d], "xiaojev_p_relevant": scores[d]} for d in ordered]
