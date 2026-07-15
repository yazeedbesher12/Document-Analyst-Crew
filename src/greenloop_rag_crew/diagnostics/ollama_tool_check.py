"""Minimal Ollama and document_search tool-calling diagnostic."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from crewai import Agent, Crew, Process, Task
from pydantic import PrivateAttr

from greenloop_rag_crew.llm import create_llm, get_llm_settings, ollama_model_name
from greenloop_rag_crew.tools.document_search import DocumentSearchTool


@dataclass
class OllamaHealth:
    reachable: bool
    model: str
    model_installed: bool
    base_url: str
    models: list[str] = field(default_factory=list)
    error_type: str | None = None
    message: str | None = None


@dataclass
class ToolCallRecord:
    query: str
    top_k: int
    document_id: str | None
    status: str
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    error_type: str | None = None


@dataclass
class SmokeResult:
    passed: bool
    response: str
    tool_calls: list[ToolCallRecord]
    errors: list[str] = field(default_factory=list)


class RecordingDocumentSearchTool(DocumentSearchTool):
    """Diagnostic-only wrapper that records CrewAI tool calls."""

    _call_records: list[ToolCallRecord] = PrivateAttr(default_factory=list)

    @property
    def call_records(self) -> list[ToolCallRecord]:
        return self._call_records

    def _run(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> str:
        output = super()._run(query=query, top_k=top_k, document_id=document_id)
        status = "error"
        pages: list[int] = []
        chunk_ids: list[str] = []
        error_type = None

        try:
            payload = json.loads(output)
            status = str(payload.get("status", "error"))
            error_type = payload.get("error_type")
            for result in payload.get("results", []):
                if isinstance(result.get("page"), int):
                    pages.append(result["page"])
                if isinstance(result.get("chunk_id"), str):
                    chunk_ids.append(result["chunk_id"])
        except json.JSONDecodeError:
            error_type = "invalid_tool_json"

        self._call_records.append(
            ToolCallRecord(
                query=query,
                top_k=top_k,
                document_id=document_id,
                status=status,
                pages=pages,
                chunk_ids=chunk_ids,
                error_type=error_type,
            )
        )
        return output


def check_ollama_health(timeout: float = 5.0) -> OllamaHealth:
    """Check Ollama /api/tags and verify qwen3:8b is installed."""

    model, base_url = get_llm_settings()
    installed_model = ollama_model_name(model)
    url = f"{base_url}/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        message = str(reason)
        error_type = "timeout" if "timed out" in message.lower() else "connection_refused"
        return OllamaHealth(
            reachable=False,
            model=installed_model,
            model_installed=False,
            base_url=base_url,
            error_type=error_type,
            message=f"Could not reach Ollama at {url}: {message}",
        )
    except TimeoutError as exc:
        return OllamaHealth(
            reachable=False,
            model=installed_model,
            model_installed=False,
            base_url=base_url,
            error_type="timeout",
            message=f"Timed out reaching Ollama at {url}: {exc}",
        )

    try:
        payload = json.loads(raw)
        models = sorted(
            item.get("name", "")
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        )
    except Exception:
        return OllamaHealth(
            reachable=True,
            model=installed_model,
            model_installed=False,
            base_url=base_url,
            error_type="invalid_response",
            message="Ollama /api/tags returned an invalid JSON response.",
        )

    installed = installed_model in models
    return OllamaHealth(
        reachable=True,
        model=installed_model,
        model_installed=installed,
        base_url=base_url,
        models=models,
        error_type=None if installed else "model_missing",
        message=None if installed else f"Ollama model {installed_model!r} is not installed.",
    )


def run_remote_policy_smoke(verbose: bool = True) -> SmokeResult:
    """Run one temporary CrewAI agent task and verify document_search was called."""

    tool = RecordingDocumentSearchTool()
    agent = Agent(
        role="Local Document Retrieval Tester",
        goal="Call document_search exactly as instructed and answer only from evidence.",
        backstory=(
            "You are testing whether local Ollama qwen3:8b can call a CrewAI custom "
            "tool. Never answer GreenLoop document questions from memory."
        ),
        llm=create_llm(),
        tools=[tool],
        allow_delegation=False,
        verbose=verbose,
        max_iter=3,
        max_retry_limit=0,
        max_execution_time=180,
    )
    task = Task(
        description=(
            "/no_think\n\n"
            "You must call document_search exactly once using:\n"
            "- query: remote work policy eligibility and number of remote days\n"
            "- document_id: HR-HBK-2025-v1.4\n"
            "- top_k: 3\n\n"
            "Then answer only from the returned evidence.\n"
            "Include the source filename, PDF page, and chunk ID.\n"
            "Keep the answer under 100 words."
        ),
        expected_output=(
            "A short evidence-grounded answer with source filename, PDF page, and chunk ID."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
    )

    try:
        response = str(crew.kickoff())
    except Exception as exc:
        message = str(exc)
        error = (
            "Local Ollama qwen3:8b request timed out before completing the tool-call smoke."
            if _looks_like_timeout(message)
            else f"CrewAI/Ollama smoke failed: {message}"
        )
        return SmokeResult(
            passed=False,
            response="",
            tool_calls=tool.call_records,
            errors=[error],
        )

    errors = _validate_smoke(tool.call_records, response)
    return SmokeResult(
        passed=not errors,
        response=response,
        tool_calls=tool.call_records,
        errors=errors,
    )


def run_diagnostic(verbose: bool = True) -> dict[str, Any]:
    """Run health check plus the single requested tool-calling smoke."""

    model, base_url = get_llm_settings()
    health = check_ollama_health(timeout=5)
    result: dict[str, Any] = {
        "configured_model": model,
        "ollama_base_url": base_url,
        "health": asdict(health),
        "smoke": None,
        "passed": False,
    }
    if not health.reachable or not health.model_installed:
        result["error"] = health.message
        return result

    smoke = run_remote_policy_smoke(verbose=verbose)
    result["smoke"] = {
        "passed": smoke.passed,
        "response": smoke.response,
        "tool_calls": [asdict(call) for call in smoke.tool_calls],
        "errors": smoke.errors,
    }
    result["passed"] = smoke.passed
    if not smoke.passed:
        result["error"] = "; ".join(smoke.errors)
    return result


def _validate_smoke(records: list[ToolCallRecord], response: str) -> list[str]:
    errors: list[str] = []
    if not records:
        errors.append("No document_search tool call was recorded.")
        return errors
    if len(records) > 1:
        errors.append(f"Expected one document_search call, recorded {len(records)}.")
    first = records[0]
    if first.status != "ok":
        errors.append(f"document_search returned status {first.status}.")
    if first.query != "remote work policy eligibility and number of remote days":
        errors.append(f"Unexpected tool query: {first.query!r}.")
    if first.document_id != "HR-HBK-2025-v1.4":
        errors.append(f"Unexpected document_id: {first.document_id!r}.")
    if first.top_k != 3:
        errors.append(f"Unexpected top_k: {first.top_k}.")
    if 6 not in first.pages:
        errors.append("Handbook page 6 was not retrieved.")
    lowered = response.lower()
    if "3" not in response and "three" not in lowered:
        errors.append("Final response did not state up to 3 remote days.")
    if not (".pdf" in lowered and "page" in lowered and re.search(r"_p\d{2,}_c\d{2}", response)):
        errors.append("Final response did not include source, page, and chunk ID citation.")
    return errors


def _looks_like_timeout(message: str) -> bool:
    lowered = message.lower()
    return "timed out" in lowered or "timeout" in lowered or "readtimeout" in lowered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_diagnostic(verbose=not args.quiet and not args.json)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        _print_human(result)
    if not result.get("passed"):
        raise SystemExit(1)


def _print_human(result: dict[str, Any]) -> None:
    health = result["health"]
    print(f"Ollama reachable: {'yes' if health['reachable'] else 'no'}")
    print(f"Configured model: {result['configured_model']}")
    print(f"Ollama base URL: {result['ollama_base_url']}")
    print(f"Model installed: {'yes' if health['model_installed'] else 'no'}")
    smoke = result.get("smoke")
    if smoke:
        print(f"Recorded tool-call count: {len(smoke['tool_calls'])}")
        for call in smoke["tool_calls"]:
            print(
                "Tool call: "
                f"query={call['query']!r}, top_k={call['top_k']}, "
                f"document_id={call['document_id']!r}, status={call['status']}, "
                f"pages={call['pages']}, chunk_ids={call['chunk_ids']}"
            )
        print(f"Final response: {smoke['response']}")
        print(f"Smoke result: {'pass' if smoke['passed'] else 'fail'}")
        for error in smoke["errors"]:
            print(f"Error: {error}", file=sys.stderr)
    if result.get("error") and not smoke:
        print(f"Error: {result['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
