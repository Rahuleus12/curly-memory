"""
Consensus Simulator — Simulate diverse people reaching consensus on questions.

This package provides tools to:
- Create diverse AI agents with unique personalities, backgrounds, and temperature settings
- Run multi-round discussions where agents respond to questions and each other
- Detect and measure consensus across simulated participants
- Generate detailed transcripts and summary reports

Quick start::

    from consensus_simulator import quick_simulate

    result = quick_simulate(
        question="Should AI be regulated by governments?",
        num_agents=5,
        max_rounds=4,
    )
"""

__version__ = "1.0.0"
__author__ = "Consensus Simulator Contributors"

# High-level convenience function
# Agent classes
from consensus_simulator.agents import (
    Agent,
    AgentConfig,
    AgentFactory,
    AgentProfile,
    EducationLevel,
    ThinkingStyle,
)

# Prompt templates
from consensus_simulator.prompts import (
    PERSONA_TEMPLATES,
    build_initial_prompt,
    build_round_prompt,
    build_system_prompt,
    get_all_persona_names,
    get_persona_by_index,
    get_personas_by_indices,
    get_random_personas,
)

# Core engine and data types
from consensus_simulator.simulation import (
    RoundResponse,
    RoundResult,
    SimulationEngine,
    SimulationResult,
    quick_simulate,
)

# Utility functions
from consensus_simulator.utils import (
    build_summary_markdown,
    build_transcript_markdown,
    calculate_average_sentiment,
    calculate_consensus_level,
    classify_stance,
    extract_stance_tag,
    format_agent_response,
    format_divider,
    format_round_header,
    group_stances,
    load_json,
    safe_filename,
    save_json,
    save_markdown,
    stance_to_numeric,
    timestamp_str,
)

__all__ = [
    # Version
    "__version__",
    # Convenience
    "quick_simulate",
    # Engine
    "SimulationEngine",
    "SimulationResult",
    "RoundResult",
    "RoundResponse",
    # Agents
    "Agent",
    "AgentFactory",
    "AgentProfile",
    "AgentConfig",
    "ThinkingStyle",
    "EducationLevel",
    # Prompts
    "build_system_prompt",
    "build_initial_prompt",
    "build_round_prompt",
    "get_persona_by_index",
    "get_all_persona_names",
    "get_personas_by_indices",
    "get_random_personas",
    "PERSONA_TEMPLATES",
    # Utils
    "extract_stance_tag",
    "classify_stance",
    "format_agent_response",
    "format_round_header",
    "format_divider",
    "build_transcript_markdown",
    "build_summary_markdown",
    "calculate_consensus_level",
    "group_stances",
    "stance_to_numeric",
    "calculate_average_sentiment",
    "save_json",
    "load_json",
    "save_markdown",
    "timestamp_str",
    "safe_filename",
]
