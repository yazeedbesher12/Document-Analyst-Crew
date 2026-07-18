from pathlib import Path

from dotenv import load_dotenv

from greenloop_rag_crew.crew import create_crew


def main() -> None:
    load_dotenv()

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    output_path = Path("output/manual_report.md")

    print("\nRunning the GreenLoop crew...")
    print("This may take several minutes with qwen3:8b.\n")

    bundle = create_crew(
        output_path=output_path,
    )

    result = bundle.crew.kickoff(
        inputs={
            "question": question,
        }
    )

    print("\nFinal result:\n")
    print(result.raw)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()