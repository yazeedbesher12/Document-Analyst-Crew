"""Standalone hybrid dense plus BM25 retrieval CLI."""

from __future__ import annotations

import argparse
import json
import sys

from greenloop_rag_crew.rag.hybrid_retriever import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CANDIDATE_K,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_RRF_K,
    HybridRetriever,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=DEFAULT_BM25_WEIGHT)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        retriever = HybridRetriever()
        results = retriever.search(
            args.query,
            top_k=args.top_k,
            document_id=args.document_id,
            candidate_k=args.candidate_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            rrf_k=args.rrf_k,
        )
    except Exception as exc:
        print(f"Hybrid search failed: {exc}", file=sys.stderr)
        raise

    if args.json:
        print(json.dumps([result.model_dump() for result in results], ensure_ascii=False))
        return

    for result in results:
        preview = " ".join(result.text.split())[:240]
        dense = (
            f"dense_rank={result.dense_rank} dense_score={result.dense_score:.4f}"
            if result.dense_rank is not None and result.dense_score is not None
            else "dense_rank=- dense_score=-"
        )
        bm25 = (
            f"bm25_rank={result.bm25_rank} bm25_score={result.bm25_score:.4f}"
            if result.bm25_rank is not None and result.bm25_score is not None
            else "bm25_rank=- bm25_score=-"
        )
        print(
            f"{result.rank}. fusion={result.fusion_score:.6f} "
            f"matched_by={','.join(result.matched_by)} {dense} {bm25}"
        )
        print(
            f"   source={result.source} page={result.page} "
            f"section={result.section}"
        )
        print(f"   chunk_id={result.chunk_id}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
