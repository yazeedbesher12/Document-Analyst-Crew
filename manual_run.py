import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from greenloop_rag_crew.crew import create_crew


def create_output_path(question: str, output_dir: Path = Path("output")) -> Path:
    """Return a unique, readable report path for one manual question."""

    slug = re.sub(r"[^\w]+", "-", question.lower(), flags=re.UNICODE).strip("-")
    slug = slug[:60].rstrip("-") or "question"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid4().hex[:8]
    return output_dir / f"manual_report_{timestamp}_{slug}_{unique_id}.md"


def main() -> None:
    load_dotenv()

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    output_path = create_output_path(question)

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
