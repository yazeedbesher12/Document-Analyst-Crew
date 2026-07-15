"""Validate CrewAI crew construction without contacting Ollama."""

from __future__ import annotations

from crewai import LLM, Process

from greenloop_rag_crew.crew import create_crew


def main() -> None:
    fake_llm = LLM(
        model="ollama/qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.6,
        top_p=0.95,
        timeout=1,
        max_tokens=1000,
    )
    bundle = create_crew(llm=fake_llm, output_path="output/crew_check_report.md")
    agents = [
        bundle.agents.document_researcher,
        bundle.agents.fact_checker,
        bundle.agents.report_writer,
    ]
    tasks = [
        ("research_task", bundle.tasks.research_task),
        ("fact_check_task", bundle.tasks.fact_check_task),
        ("report_task", bundle.tasks.report_task),
    ]

    print(f"Agent order: {[agent.role for agent in agents]}")
    print(f"Task order: {[name for name, _task in tasks]}")
    for name, task in tasks:
        context_names = [
            context_task.description[:40].replace("\n", " ")
            for context_task in (task.context or [])
        ]
        print(f"{name}:")
        print(f"  agent: {_agent_key(task.agent.role)}")
        print(f"  context_count: {len(task.context or [])}")
        print(f"  context: {context_names}")
    print(f"Process: {bundle.crew.process.value}")
    print(f"Report output path: {bundle.tasks.report_task.output_file}")

    tool_assignments = {
        agent.role: [tool.name for tool in agent.tools]
        for agent in agents
    }
    print(f"Tool assignments: {tool_assignments}")
    web_tools = [
        tool_name
        for tools in tool_assignments.values()
        for tool_name in tools
        if tool_name != "document_search"
    ]
    print(f"Web tools: {str(bool(web_tools)).lower()}")

    passed = (
        [agent.role for agent in agents]
        == ["Document Researcher", "Fact Checker", "Report Writer"]
        and [name for name, _task in tasks]
        == ["research_task", "fact_check_task", "report_task"]
        and bundle.tasks.research_task.agent is bundle.agents.document_researcher
        and bundle.tasks.fact_check_task.agent is bundle.agents.fact_checker
        and bundle.tasks.report_task.agent is bundle.agents.report_writer
        and bundle.tasks.research_task.context == []
        and bundle.tasks.fact_check_task.context == [bundle.tasks.research_task]
        and bundle.tasks.report_task.context
        == [bundle.tasks.research_task, bundle.tasks.fact_check_task]
        and bundle.crew.process == Process.sequential
        and tool_assignments["Report Writer"] == []
        and not web_tools
    )
    print(f"Result: {'pass' if passed else 'fail'}")
    if not passed:
        raise SystemExit(1)


def _agent_key(role: str) -> str:
    return {
        "Document Researcher": "document_researcher",
        "Fact Checker": "fact_checker",
        "Report Writer": "report_writer",
    }.get(role, role)


if __name__ == "__main__":
    main()
