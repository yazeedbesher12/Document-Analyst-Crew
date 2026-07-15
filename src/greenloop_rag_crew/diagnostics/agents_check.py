"""Validate final agent construction without contacting Ollama."""

from __future__ import annotations

from crewai import LLM

from greenloop_rag_crew.agents import create_agents


def main() -> None:
    fake_llm = LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.6,
        top_p=0.95,
        timeout=1,
        max_tokens=1000,
    )
    bundle = create_agents(llm=fake_llm)
    agents = {
        "document_researcher": bundle.document_researcher,
        "fact_checker": bundle.fact_checker,
        "report_writer": bundle.report_writer,
    }

    print(f"Agent count: {len(agents)}")
    for key, agent in agents.items():
        print(f"{key}:")
        print(f"  role: {agent.role}")
        print(f"  tools: {[tool.name for tool in agent.tools]}")
        print(f"  max_iter: {agent.max_iter}")
        print(f"  allow_delegation: {agent.allow_delegation}")

    researcher_tool = bundle.document_researcher.tools[0]
    checker_tool = bundle.fact_checker.tools[0]
    shared_tool = researcher_tool is checker_tool is bundle.document_search_tool
    report_writer_has_no_tools = len(bundle.report_writer.tools) == 0
    web_tools = [
        tool.name
        for agent in agents.values()
        for tool in agent.tools
        if tool.name != "document_search"
    ]

    print(f"Shared document_search: {str(shared_tool).lower()}")
    print(f"Report Writer has zero tools: {str(report_writer_has_no_tools).lower()}")
    print(f"Web tools: {str(bool(web_tools)).lower()}")

    passed = (
        len(agents) == 3
        and shared_tool
        and report_writer_has_no_tools
        and not web_tools
    )
    print(f"Result: {'pass' if passed else 'fail'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
