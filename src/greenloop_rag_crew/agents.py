"""CrewAI Agent construction for the final GreenLoop RAG roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crewai import Agent

from greenloop_rag_crew.config_loader import (
    DOCUMENT_SEARCH_TOOL,
    AgentConfig,
    ConfigError,
    load_agents_config,
)
from greenloop_rag_crew.llm import create_llm
from greenloop_rag_crew.tools import DocumentSearchTool

TOOL_REGISTRY = {
    DOCUMENT_SEARCH_TOOL: DocumentSearchTool,
}


@dataclass(frozen=True)
class AgentBundle:
    """Final agent objects plus the shared document_search tool."""

    document_researcher: Agent
    fact_checker: Agent
    report_writer: Agent
    document_search_tool: DocumentSearchTool


def create_agents(llm=None, config_path: str | Path | None = None) -> AgentBundle:
    """Construct the three final CrewAI agents from agents.jsonc."""

    config = load_agents_config(config_path=config_path)
    shared_llm = llm if llm is not None else create_llm()
    document_search_tool = DocumentSearchTool()

    return AgentBundle(
        document_researcher=_create_agent(
            config.document_researcher,
            llm=shared_llm,
            document_search_tool=document_search_tool,
        ),
        fact_checker=_create_agent(
            config.fact_checker,
            llm=shared_llm,
            document_search_tool=document_search_tool,
        ),
        report_writer=_create_agent(
            config.report_writer,
            llm=shared_llm,
            document_search_tool=document_search_tool,
        ),
        document_search_tool=document_search_tool,
    )


def _create_agent(
    agent_config: AgentConfig,
    llm,
    document_search_tool: DocumentSearchTool,
) -> Agent:
    tools = _resolve_tools(agent_config.tools, document_search_tool)
    return Agent(
        role=agent_config.role,
        goal=agent_config.goal,
        backstory=agent_config.backstory,
        tools=tools,
        allow_delegation=agent_config.allow_delegation,
        verbose=agent_config.verbose,
        max_iter=agent_config.max_iter,
        llm=llm,
    )


def _resolve_tools(
    tool_names: list[str],
    document_search_tool: DocumentSearchTool,
) -> list[DocumentSearchTool]:
    resolved_tools: list[DocumentSearchTool] = []
    for tool_name in tool_names:
        tool_class = TOOL_REGISTRY.get(tool_name)
        if tool_class is None:
            raise ConfigError(
                f"Unknown tool {tool_name!r}. Allowed tools: {', '.join(TOOL_REGISTRY)}"
            )
        if tool_name == DOCUMENT_SEARCH_TOOL:
            resolved_tools.append(document_search_tool)
    return resolved_tools
