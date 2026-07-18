"""Command-line runner for GreenLoop RAG report questions."""

from __future__ import annotations

import argparse
import json

from greenloop_rag_crew.runner import list_questions, run_all, run_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List configured questions.")
    parser.add_argument("--question-id", help="Run or dry-run one configured question.")
    parser.add_argument("--all", action="store_true", help="Run or dry-run all questions.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without kickoff.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing reports.")
    args = parser.parse_args()

    _validate_args(args)

    if args.list:
        payload = [
            {
                "id": question.id,
                "question": question.question,
                "output_file": question.output_file,
                "required_document_ids": question.required_document_ids,
            }
            for question in list_questions()
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.question_id:
        result = run_question(
            args.question_id,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str))
        return

    results = run_all(overwrite=args.overwrite, dry_run=args.dry_run)
    print(
        json.dumps(
            [result.__dict__ for result in results],
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def _validate_args(args: argparse.Namespace) -> None:
    selected = sum(bool(value) for value in [args.list, args.question_id, args.all])
    if selected != 1:
        raise SystemExit("Choose exactly one of --list, --question-id, or --all.")
    if args.list and (args.dry_run or args.overwrite):
        raise SystemExit("--list cannot be combined with --dry-run or --overwrite.")


if __name__ == "__main__":
    main()
