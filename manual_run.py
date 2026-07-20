from greenloop_rag_crew.question_execution import (
    QuestionExecutionError,
    QuestionValidationError,
    create_unique_report_path,
    execute_question,
)

create_output_path = create_unique_report_path


def main() -> None:
    question = input("Enter your question: ").strip()

    print("\nRunning the GreenLoop crew...")
    print("This may take several minutes with qwen3:8b.\n")

    try:
        result = execute_question(question)
    except QuestionValidationError as exc:
        print(str(exc))
        return
    except QuestionExecutionError as exc:
        print(str(exc))
        return

    print("\nFinal result:\n")
    print(result.report_markdown)
    print(f"\nReport saved to: {result.output_path}")


if __name__ == "__main__":
    main()
