"""
Configuration module for the Consensus Simulator.

Defines Pydantic models for agent profiles, simulation settings,
and question configurations.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PersonalityArchetype(str, Enum):
    """Broad personality archetypes that influence agent behavior."""

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PRAGMATIC = "pragmatic"
    EMPATHETIC = "empathetic"
    SKEPTICAL = "skeptical"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"
    PROGRESSIVE = "progressive"
    DIPLOMATIC = "diplomatic"
    DIRECT = "direct"


class EducationLevel(str, Enum):
    """Education levels for agent backgrounds."""

    HIGH_SCHOOL = "high_school"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    DOCTORATE = "doctorate"
    SELF_TAUGHT = "self_taught"


class PoliticalLean(str, Enum):
    """Political leaning of an agent (optional context)."""

    LIBERAL = "liberal"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    LIBERTARIAN = "libertarian"
    APOLITICAL = "apolitical"


# ---------------------------------------------------------------------------
# Agent Profile
# ---------------------------------------------------------------------------


class AgentProfile(BaseModel):
    """
    Full definition of a simulated person / agent.

    Each field contributes to the system prompt that shapes how the LLM
    responds during the simulation.
    """

    name: str = Field(
        ...,
        description="Human-readable name for the agent.",
    )
    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age of the simulated person.",
    )
    occupation: str = Field(
        ...,
        description="Current or former occupation.",
    )
    education: EducationLevel = Field(
        default=EducationLevel.BACHELORS,
        description="Highest level of education completed.",
    )
    personality: PersonalityArchetype = Field(
        default=PersonalityArchetype.PRAGMATIC,
        description="Dominant personality archetype.",
    )
    political_lean: Optional[PoliticalLean] = Field(
        default=None,
        description="Optional political leaning (leave None for neutral).",
    )
    background_summary: str = Field(
        default="",
        description="A short paragraph describing the agent's life story and worldview.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "LLM sampling temperature. Lower values make the agent more "
            "deterministic and consistent; higher values make it more varied "
            "and creative."
        ),
    )
    persuasion_resistance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How resistant this agent is to changing their opinion. "
            "0.0 = easily swayed, 1.0 = very stubborn."
        ),
    )
    verbosity: str = Field(
        default="moderate",
        description=(
            "How verbose the agent's responses should be: "
            "'brief', 'moderate', or 'detailed'."
        ),
    )

    @field_validator("verbosity")
    @classmethod
    def validate_verbosity(cls, v: str) -> str:
        allowed = {"brief", "moderate", "detailed"}
        if v.lower() not in allowed:
            raise ValueError(f"verbosity must be one of {allowed}, got '{v}'")
        return v.lower()

    def build_system_prompt(self, question: str) -> str:
        """
        Build the system prompt that will be sent to the LLM for this agent.
        """
        political_str = (
            f"Politically, you lean {self.political_lean.value}."
            if self.political_lean
            else "You do not have a strong political affiliation."
        )

        verbosity_instruction = {
            "brief": "Keep your responses concise — 2-3 sentences at most.",
            "moderate": "Respond in a moderate length — around 1 paragraph.",
            "detailed": "Provide detailed, thorough responses — multiple paragraphs if needed.",
        }.get(self.verbosity, "Respond in a moderate length.")

        resistance_instruction = ""
        if self.persuasion_resistance >= 0.7:
            resistance_instruction = (
                "You are very stubborn and reluctant to change your opinion. "
                "You need very strong arguments to be convinced."
            )
        elif self.persuasion_resistance <= 0.3:
            resistance_instruction = (
                "You are open-minded and easily swayed by good arguments. "
                "You are willing to change your position."
            )
        else:
            resistance_instruction = (
                "You are moderately open to changing your opinion if presented "
                "with compelling reasoning."
            )

        prompt = (
            f"You are {self.name}, a {self.age}-year-old {self.occupation.lower()}. "
            f"You have a {self.education.value.replace('_', ' ')} education. "
            f"Your dominant personality trait is {self.personality.value}. "
            f"{political_str}\n\n"
            f"Background: {self.background_summary}\n\n"
            f"Behavioral notes:\n"
            f"- {resistance_instruction}\n"
            f"- {verbosity_instruction}\n"
            f"- Stay in character at all times. Express opinions as {self.name} would.\n"
            f"- Be authentic — it's okay to disagree with others or express uncertainty.\n"
            f"- Base your responses on your background, experience, and personality.\n\n"
            f'The question under discussion is:\n"{question}"\n\n'
            f"Respond as {self.name} would in a group discussion setting."
        )
        return prompt


# ---------------------------------------------------------------------------
# Simulation Settings
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for the LLM model used in the simulation."""

    api_key: str = Field(
        default="",
        description="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible API.",
    )
    model_name: str = Field(
        default="gpt-4",
        description="Model identifier to use (e.g. gpt-4, gpt-3.5-turbo).",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum tokens per response.",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=5,
        description="Request timeout in seconds.",
    )

    def get_api_key(self) -> str:
        """Return the API key, falling back to the environment variable."""
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if not env_key:
            raise ValueError(
                "No API key provided. Set api_key in config or the "
                "OPENAI_API_KEY environment variable."
            )
        return env_key


class SimulationSettings(BaseModel):
    """Top-level settings for a simulation run."""

    max_rounds: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum discussion rounds before forcing a conclusion.",
    )
    consensus_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description=(
            "Fraction of agents that must agree for consensus to be declared. "
            "0.7 means 70% of agents must share the same position."
        ),
    )
    parallel_requests: bool = Field(
        default=True,
        description="Whether to issue LLM requests in parallel for speed.",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging to the console.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory where results are saved.",
    )
    save_transcript: bool = Field(
        default=True,
        description="Save the full discussion transcript to a file.",
    )
    save_summary: bool = Field(
        default=True,
        description="Save a summary report after the simulation.",
    )


# ---------------------------------------------------------------------------
# Question Configuration
# ---------------------------------------------------------------------------


class QuestionConfig(BaseModel):
    """A question to be debated by the simulated agents."""

    text: str = Field(
        ...,
        description="The question or topic for discussion.",
    )
    context: str = Field(
        default="",
        description="Additional context or background information for the question.",
    )
    category: str = Field(
        default="general",
        description="Category label (e.g. 'ethics', 'policy', 'technology').",
    )
    expected_stances: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of expected possible stances (e.g. 'for', 'against'). "
            "Used to help classify agent positions."
        ),
    )

    def build_question_prompt(self) -> str:
        """Build the full question prompt including any context."""
        prompt = f"Question: {self.text}"
        if self.context:
            prompt += f"\n\nContext: {self.context}"
        return prompt


# ---------------------------------------------------------------------------
# Full Simulation Configuration
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """
    Complete configuration for a simulation run.

    Combines model settings, simulation parameters, agent profiles,
    and the question(s) to be discussed.
    """

    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="LLM model configuration.",
    )
    settings: SimulationSettings = Field(
        default_factory=SimulationSettings,
        description="Simulation parameters.",
    )
    agents: list[AgentProfile] = Field(
        default_factory=list,
        description="List of agent profiles to simulate.",
    )
    questions: list[QuestionConfig] = Field(
        default_factory=list,
        description="Questions to be discussed.",
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "SimulationConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        """Create configuration from a dictionary."""
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)


# ---------------------------------------------------------------------------
# Default Agent Presets
# ---------------------------------------------------------------------------


def get_default_agents() -> list[AgentProfile]:
    """
    Return a diverse set of default agent profiles suitable for
    out-of-the-box simulation.
    """
    return [
        AgentProfile(
            name="Dr. Alice Chen",
            age=45,
            occupation="University Professor of Ethics",
            education=EducationLevel.DOCTORATE,
            personality=PersonalityArchetype.ANALYTICAL,
            political_lean=PoliticalLean.MODERATE,
            background_summary=(
                "Alice has spent 20 years studying moral philosophy and applied "
                "ethics. She values logical consistency and carefully considers "
                "all perspectives before forming an opinion."
            ),
            temperature=0.4,
            persuasion_resistance=0.6,
            verbosity="detailed",
        ),
        AgentProfile(
            name="Marcus Johnson",
            age=32,
            occupation="Software Engineer",
            education=EducationLevel.BACHELORS,
            personality=PersonalityArchetype.PRAGMATIC,
            political_lean=PoliticalLean.LIBERAL,
            background_summary=(
                "Marcus works at a tech startup and is passionate about "
                "innovation. He tends to favor practical solutions over "
                "idealistic ones and values efficiency."
            ),
            temperature=0.6,
            persuasion_resistance=0.4,
            verbosity="moderate",
        ),
        AgentProfile(
            name="Elena Rodriguez",
            age=58,
            occupation="Retired Nurse",
            education=EducationLevel.MASTERS,
            personality=PersonalityArchetype.EMPATHETIC,
            political_lean=PoliticalLean.MODERATE,
            background_summary=(
                "Elena spent 30 years working in hospitals and community "
                "clinics. She has seen the human side of policy decisions and "
                "deeply cares about people's well-being."
            ),
            temperature=0.7,
            persuasion_resistance=0.5,
            verbosity="moderate",
        ),
        AgentProfile(
            name="James Wright",
            age=67,
            occupation="Former Business Executive",
            education=EducationLevel.MASTERS,
            personality=PersonalityArchetype.CONSERVATIVE,
            political_lean=PoliticalLean.CONSERVATIVE,
            background_summary=(
                "James built a successful manufacturing company over 35 years. "
                "He believes strongly in free markets, personal responsibility, "
                "and tradition. He is skeptical of rapid change."
            ),
            temperature=0.5,
            persuasion_resistance=0.8,
            verbosity="moderate",
        ),
        AgentProfile(
            name="Priya Sharma",
            age=28,
            occupation="Environmental Activist",
            education=EducationLevel.BACHELORS,
            personality=PersonalityArchetype.PROGRESSIVE,
            political_lean=PoliticalLean.LIBERAL,
            background_summary=(
                "Priya has been involved in climate activism since college. "
                "She is passionate about social justice and environmental "
                "issues. She is not afraid to challenge the status quo."
            ),
            temperature=0.9,
            persuasion_resistance=0.7,
            verbosity="detailed",
        ),
        AgentProfile(
            name="Tom O'Brien",
            age=41,
            occupation="Small Business Owner",
            education=EducationLevel.HIGH_SCHOOL,
            personality=PersonalityArchetype.SKEPTICAL,
            political_lean=PoliticalLean.LIBERTARIAN,
            background_summary=(
                "Tom runs a family-owned hardware store. He values his "
                "independence and is suspicious of government overreach. "
                "He trusts his own experience over expert opinions."
            ),
            temperature=0.8,
            persuasion_resistance=0.6,
            verbosity="brief",
        ),
        AgentProfile(
            name="Dr. Yuki Tanaka",
            age=36,
            occupation="Data Scientist",
            education=EducationLevel.DOCTORATE,
            personality=PersonalityArchetype.ANALYTICAL,
            political_lean=PoliticalLean.APOLITICAL,
            background_summary=(
                "Yuki specializes in statistical modeling and loves working "
                "with numbers. She tries to remain objective and data-driven "
                "in all her assessments, avoiding emotional reasoning."
            ),
            temperature=0.3,
            persuasion_resistance=0.5,
            verbosity="detailed",
        ),
        AgentProfile(
            name="Fatima Al-Hassan",
            age=52,
            occupation="Community Organizer",
            education=EducationLevel.BACHELORS,
            personality=PersonalityArchetype.DIPLOMATIC,
            political_lean=PoliticalLean.PROGRESSIVE,
            background_summary=(
                "Fatima has spent decades building bridges between diverse "
                "communities. She excels at finding common ground and "
                "mediating disagreements. She values harmony and collective "
                "well-being."
            ),
            temperature=0.6,
            persuasion_resistance=0.3,
            verbosity="moderate",
        ),
    ]


def get_sample_questions() -> list[QuestionConfig]:
    """Return a list of sample questions for testing."""
    return [
        QuestionConfig(
            text="Should artificial intelligence be regulated by governments?",
            context=(
                "AI systems are becoming increasingly powerful and autonomous. "
                "Some argue regulation stifles innovation while others believe "
                "it's essential for public safety."
            ),
            category="technology",
            expected_stances=[
                "strongly for",
                "somewhat for",
                "neutral",
                "somewhat against",
                "strongly against",
            ],
        ),
        QuestionConfig(
            text="Is a four-day work week beneficial for society?",
            context=(
                "Several countries and companies have experimented with a "
                "four-day work week. Results have been mixed, with some "
                "reporting increased productivity and others citing challenges."
            ),
            category="policy",
            expected_stances=[
                "strongly for",
                "somewhat for",
                "neutral",
                "somewhat against",
                "strongly against",
            ],
        ),
        QuestionConfig(
            text="Should universal basic income be implemented nationwide?",
            context=(
                "UBI proposals vary, but generally involve providing all "
                "citizens with a regular, unconditional sum of money. "
                "Proponents argue it reduces poverty; critics worry about "
                "costs and work disincentives."
            ),
            category="economics",
            expected_stances=[
                "strongly for",
                "somewhat for",
                "neutral",
                "somewhat against",
                "strongly against",
            ],
        ),
    ]
