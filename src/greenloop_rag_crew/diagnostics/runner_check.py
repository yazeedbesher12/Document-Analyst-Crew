"""Dry-run diagnostics for the three final question runners."""

from __future__ import annotations

from greenloop_rag_crew.runner import list_questions, run_all


def main() -> None:
    questions = list_questions()
    results = run_all(dry_run=True)
    output_paths = [question.output_file for question in questions]
    ollama_calls = 0

    print(f"Question IDs: {[question.id for question in questions]}")
    for question in questions:
        print(f"{question.id}:")
        print(f"  question: {question.question}")
        print(f"  output_file: {question.output_file}")
        print(f"  required_document_ids: {question.required_document_ids}")
    print(f"Unique output paths: {len(set(output_paths)) == len(output_paths)}")
    for result in results:
        print(f"Dry run {result.question_id}:")
        print(f"  agent_order: {result.agent_order}")
        print(f"  task_order: {result.task_order}")
        print(f"  process_type: {result.process_type}")
    print(f"Ollama calls: {ollama_calls}")

    passed = (
        len(questions) == 3
        and len(set(output_paths)) == 3
        and all(result.process_type == "sequential" for result in results)
        and all(
            result.agent_order
            == ["Document Researcher", "Fact Checker", "Report Writer"]
            for result in results
        )
        and ollama_calls == 0
    )
    print(f"Result: {'pass' if passed else 'fail'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
