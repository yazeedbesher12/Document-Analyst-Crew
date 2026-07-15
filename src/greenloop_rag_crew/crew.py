"""CrewAI Crew construction for the GreenLoop RAG Document Analyst Crew."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crewai import Crew, Process

from greenloop_rag_crew.agents import AgentBundle, create_agents
from greenloop_rag_crew.tasks import TaskBundle, create_tasks


@dataclass(frozen=True)
class CrewBundle:
    """Constructed agents, tasks, and sequential CrewAI crew."""

    agents: AgentBundle
    tasks: TaskBundle
    crew: Crew


def create_crew(llm=None, output_path: str | Path = "output/report.md") -> CrewBundle:
    """Construct the final sequential crew without kicking it off."""

    agents = create_agents(llm=llm)
    tasks = create_tasks(agents=agents, output_path=output_path)
    crew = Crew(
        agents=[
            agents.document_researcher,
            agents.fact_checker,
            agents.report_writer,
        ],
        tasks=[
            tasks.research_task,
            tasks.fact_check_task,
            tasks.report_task,
        ],
        process=Process.sequential,
        verbose=True,
    )
    return CrewBundle(agents=agents, tasks=tasks, crew=crew)
