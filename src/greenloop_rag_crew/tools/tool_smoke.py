"""Smoke CLI for the CrewAI document_search tool."""

from __future__ import annotations

import argparse

from greenloop_rag_crew.tools.document_search import DocumentSearchTool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args()

    tool = DocumentSearchTool()
    try:
        output = tool.run(
            query=args.query,
            top_k=args.top_k,
            document_id=args.document_id,
        )
    except Exception:
        output = tool._run(
            query=args.query,
            top_k=args.top_k,
            document_id=args.document_id,
        )
    print(output)


if __name__ == "__main__":
    main()
