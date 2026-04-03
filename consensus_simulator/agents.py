"""
agents.py - Agent definitions and factory for creating simulated people.

Each agent represents a unique "person" with a distinct personality,
background, and temperature setting that influences how they respond
to questions and engage in group discussions.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------


class ThinkingStyle(str, Enum):
    """Cognitive style that biases how an agent reasons."""

    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PRAGMATIC = "pragmatic"
    EMOTIONAL = "emotional"
    SKEPTICAL = "skeptical"
    OPTIMISTIC = "optimistic"
    CONSERVATIVE = "conservative"
    COLLABORATIVE = "collaborative"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    DOCTORATE = "doctorate"
    SELF_TAUGHT = "self_taught"


@dataclass
class AgentProfile:
    """Biographical and psychological profile of a simulated person."""

    name: str
    age: int
    occupation: str
    education: EducationLevel
    background: str
    personality_traits: list[str]
    thinking_style: ThinkingStyle
    values: list[str]
    communication_style: str
    bio: str


@dataclass
class AgentConfig:
    """Runtime configuration for an agent."""

    temperature: float
    max_tokens: int = 300
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class Agent:
    """
    A simulated person backed by an LLM.

    The agent has a persistent identity (profile) and configuration
    (temperature, etc.) that together shape every response it gives.
    """

    def __init__(
        self,
        profile: AgentProfile,
        config: AgentConfig,
        client: OpenAI,
        model: str = "gpt-4",
    ) -> None:
        self.profile = profile
        self.config = config
        self.client = client
        self.model = model
        self._history: list[dict[str, str]] = []

    # --- properties --------------------------------------------------------

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def thinking_style(self) -> ThinkingStyle:
        return self.profile.thinking_style

    # --- system prompt construction ----------------------------------------

    def _build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build a rich system prompt that encodes the agent's identity."""
        parts: list[str] = []

        parts.append(
            f"You are {self.profile.name}, a {self.profile.age}-year-old "
            f"{self.profile.occupation.lower()}."
        )

        parts.append(f"Education: {self.profile.education.value.replace('_', ' ')}.")

        parts.append(f"Background: {self.profile.background}")

        parts.append(f"Biographical details: {self.profile.bio}")

        traits = ", ".join(self.profile.personality_traits)
        parts.append(f"Personality traits: {traits}.")

        parts.append(
            f"Thinking style: {self.profile.thinking_style.value}. "
            "Respond in a way that reflects this cognitive approach."
        )

        values = ", ".join(self.profile.values)
        parts.append(f"Core values: {values}.")

        parts.append(f"Communication style: {self.profile.communication_style}.")

        parts.append(
            "\nIMPORTANT RULES:\n"
            "1. Stay in character at all times.\n"
            "2. Express opinions that are consistent with your background, "
            "values, and personality.\n"
            "3. Be authentic — real people are sometimes uncertain, "
            "opinionated, or ambivalent.\n"
            "4. Keep responses concise (2-5 sentences) unless the topic "
            "warrants more detail.\n"
            "5. When responding to others, acknowledge their points before "
            "adding your own.\n"
            "6. You may change your mind if presented with compelling "
            "arguments, but only if it fits your character.\n"
        )

        if context:
            parts.append(f"\nAdditional context for this discussion:\n{context}")

        return "\n\n".join(parts)

    # --- LLM interaction ---------------------------------------------------

    def respond(
        self,
        question: str,
        previous_responses: Optional[list[dict[str, str]]] = None,
        context: Optional[str] = None,
        round_number: int = 1,
        is_final: bool = False,
    ) -> str:
        """
        Generate a response from this agent.

        Parameters
        ----------
        question : str
            The topic or question under discussion.
        previous_responses : list[dict], optional
            Earlier responses in the format ``[{"name": ..., "text": ...}]``.
        context : str, optional
            Extra context injected into the system prompt.
        round_number : int
            Which round of discussion this is.
        is_final : bool
            If ``True``, the agent is asked to state a final position.

        Returns
        -------
        str
            The agent's response text.
        """
        system_prompt = self._build_system_prompt(context=context)

        # Build user message
        user_parts: list[str] = []

        if round_number == 1 and not previous_responses:
            user_parts.append(
                f"Discussion Topic: {question}\n\n"
                "Please share your initial thoughts and position on this topic."
            )
        else:
            user_parts.append(f"Discussion Topic: {question}")
            user_parts.append(f"(Round {round_number} of discussion)\n")

            if previous_responses:
                user_parts.append("Here is what others have said so far:\n")
                for resp in previous_responses:
                    user_parts.append(f"- {resp['name']}: {resp['text']}")
                user_parts.append("")

            if is_final:
                user_parts.append(
                    "This is the FINAL round. Please state your conclusive "
                    "position. Begin your response with either 'AGREE:' or "
                    "'DISAGREE:' followed by a brief summary of your position."
                )
            else:
                user_parts.append(
                    "Consider what others have said and respond with your "
                    "current thoughts. You may agree, disagree, or propose "
                    "a compromise."
                )

        user_message = "\n".join(user_parts)

        # Call the LLM
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )
            text = response.choices[0].message.content or ""
        except Exception as exc:
            text = f"[Error generating response: {exc}]"

        # Record in local history
        self._history.append(
            {
                "round": round_number,
                "question": question,
                "response": text,
                "is_final": is_final,
            }
        )

        return text.strip()

    # --- consensus helpers -------------------------------------------------

    def extract_stance(self, text: str) -> Optional[str]:
        """
        Try to extract a clear AGREE / DISAGREE stance from a final response.

        Returns
        -------
        str or None
            ``"agree"``, ``"disagree"``, or ``None`` if unclear.
        """
        upper = text.upper().strip()
        if upper.startswith("AGREE"):
            return "agree"
        if upper.startswith("DISAGREE"):
            return "disagree"

        # Fallback: look for keywords
        agree_keywords = [
            "i agree",
            "i support",
            "i'm in favor",
            "i'm for",
            "consensus",
            "common ground",
            "compromise",
        ]
        disagree_keywords = [
            "i disagree",
            "i oppose",
            "i'm against",
            "i reject",
            "cannot agree",
            "do not support",
        ]

        lower = text.lower()
        for kw in agree_keywords:
            if kw in lower:
                return "agree"
        for kw in disagree_keywords:
            if kw in lower:
                return "disagree"

        return None

    # --- representation ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.profile.name!r}, "
            f"age={self.profile.age}, "
            f"occupation={self.profile.occupation!r}, "
            f"temp={self.config.temperature})"
        )

    def __str__(self) -> str:
        return (
            f"{self.profile.name} ({self.profile.age}, "
            f"{self.profile.occupation}) — "
            f"thinking: {self.profile.thinking_style.value}, "
            f"temp: {self.config.temperature}"
        )


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

# Pool of realistic first names
_FIRST_NAMES = [
    "Alice",
    "Bob",
    "Carlos",
    "Diana",
    "Ethan",
    "Fatima",
    "George",
    "Hannah",
    "Ibrahim",
    "Julia",
    "Kevin",
    "Luna",
    "Marcus",
    "Nadia",
    "Oliver",
    "Priya",
    "Quinn",
    "Rachel",
    "Samuel",
    "Tara",
    "Uma",
    "Victor",
    "Wendy",
    "Yuki",
    "Zara",
]

# Pool of occupations
_OCCUPATIONS = [
    "Software Engineer",
    "Teacher",
    "Doctor",
    "Nurse",
    "Accountant",
    "Artist",
    "Chef",
    "Journalist",
    "Lawyer",
    "Mechanic",
    "Social Worker",
    "Farmer",
    "Pharmacist",
    "Architect",
    "Electrician",
    "Marketing Manager",
    "Research Scientist",
    "Police Officer",
    "Librarian",
    "Entrepreneur",
    "Truck Driver",
    "Graphic Designer",
    "Financial Analyst",
    "Civil Engineer",
    "Veterinarian",
]

# Pool of personality traits
_PERSONALITY_TRAITS = [
    "outgoing",
    "reserved",
    "detail-oriented",
    "big-picture thinker",
    "empathetic",
    "stubborn",
    "open-minded",
    "cautious",
    "bold",
    "patient",
    "impatient",
    "diplomatic",
    "direct",
    "humorous",
    "serious",
    "idealistic",
    "realistic",
    "curious",
    "loyal",
    "independent",
    "team-oriented",
    "perfectionist",
    "adaptable",
]

# Pool of core values
_VALUES = [
    "honesty",
    "fairness",
    "freedom",
    "security",
    "tradition",
    "innovation",
    "community",
    "individuality",
    "sustainability",
    "efficiency",
    "compassion",
    "justice",
    "loyalty",
    "creativity",
    "stability",
    "progress",
    "equality",
    "meritocracy",
    "family",
    "adventure",
    "knowledge",
    "spirituality",
    "health",
    "wealth",
]

# Pool of communication styles
_COMMUNICATION_STYLES = [
    "Speaks calmly and deliberately, choosing words carefully",
    "Tends to be passionate and animated when discussing topics",
    "Prefers to listen first, then offer measured opinions",
    "Uses analogies and stories to make points",
    "Very direct and to the point, dislikes beating around the bush",
    "Diplomatic, always tries to find common ground",
    "Asks many questions before forming an opinion",
    "Speaks with conviction, rarely shows uncertainty",
    "Thoughtful, often plays devil's advocate",
    "Casual and conversational, avoids jargon",
]

# Pool of background stories
_BACKGROUNDS = [
    "Grew up in a small town and moved to the city for work.",
    "First-generation college graduate in the family.",
    "Grew up in a multicultural household with diverse perspectives.",
    "Has lived in three different countries and values global perspectives.",
    "Comes from a long line of family business owners.",
    "Raised by a single parent, learned the value of hard work early.",
    "Grew up in a rural farming community with strong traditional values.",
    "Worked in multiple industries before finding their current career.",
    "Overcame significant personal challenges that shaped their worldview.",
    "Has always been deeply involved in local community service.",
    "Immigrated at a young age and navigated between two cultures.",
    "Grew up in a household that valued education above all else.",
    "Spent several years volunteering abroad in developing nations.",
    "Was a late bloomer who found their calling later in life.",
    "Comes from a family of public servants and community leaders.",
]


class AgentFactory:
    """
    Factory for creating diverse groups of simulated agents.

    Each agent is given a unique profile and a temperature setting
    that together produce varied, realistic responses.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4",
        base_config: Optional[AgentConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.base_config = base_config or AgentConfig(temperature=0.7)
        self._rng = random.Random(seed)

    # --- helpers -----------------------------------------------------------

    def _pick(self, pool: list, count: int) -> list:
        """Pick *count* unique items from *pool*."""
        return self._rng.sample(pool, min(count, len(pool)))

    def _generate_bio(self, profile: "AgentProfile") -> str:
        """Create a short biographical blurb from the profile."""
        edu = profile.education.value.replace("_", " ")
        traits = ", ".join(profile.personality_traits[:3])
        return (
            f"A {profile.age}-year-old {profile.occupation.lower()} with "
            f"a {edu} background. Known for being {traits}. "
            f"{profile.background} Approaches problems with a "
            f"{profile.thinking_style.value} mindset."
        )

    # --- public API --------------------------------------------------------

    def create_agent(
        self,
        name: Optional[str] = None,
        age: Optional[int] = None,
        occupation: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking_style: Optional[ThinkingStyle] = None,
        education: Optional[EducationLevel] = None,
        background: Optional[str] = None,
        personality_traits: Optional[list[str]] = None,
        values: Optional[list[str]] = None,
        communication_style: Optional[str] = None,
    ) -> Agent:
        """Create a single agent with the given or random attributes."""

        _name = name or self._rng.choice(_FIRST_NAMES)
        _age = age or self._rng.randint(22, 68)
        _occupation = occupation or self._rng.choice(_OCCUPATIONS)
        _education = education or self._rng.choice(list(EducationLevel))
        _thinking = thinking_style or self._rng.choice(list(ThinkingStyle))
        _traits = personality_traits or self._pick(
            _PERSONALITY_TRAITS, self._rng.randint(2, 4)
        )
        _values = values or self._pick(_VALUES, self._rng.randint(2, 4))
        _bg = background or self._rng.choice(_BACKGROUNDS)
        _comm = communication_style or self._rng.choice(_COMMUNICATION_STYLES)

        profile = AgentProfile(
            name=_name,
            age=_age,
            occupation=_occupation,
            education=_education,
            background=_bg,
            personality_traits=_traits,
            thinking_style=_thinking,
            values=_values,
            communication_style=_comm,
            bio="",  # filled below
        )
        profile.bio = self._generate_bio(profile)

        config = AgentConfig(
            temperature=temperature
            if temperature is not None
            else self.base_config.temperature,
            max_tokens=self.base_config.max_tokens,
            top_p=self.base_config.top_p,
            frequency_penalty=self.base_config.frequency_penalty,
            presence_penalty=self.base_config.presence_penalty,
        )

        return Agent(
            profile=profile,
            config=config,
            client=self.client,
            model=self.model,
        )

    def create_group(
        self,
        count: int = 5,
        temperature_range: tuple[float, float] = (0.3, 1.2),
        ensure_diversity: bool = True,
    ) -> list[Agent]:
        """
        Create a diverse group of agents.

        Parameters
        ----------
        count : int
            Number of agents to create.
        temperature_range : tuple
            ``(min_temp, max_temp)`` — agents will be spread across this range.
        ensure_diversity : bool
            If ``True``, guarantee that each agent gets a unique name,
            occupation, and thinking style as far as the pools allow.

        Returns
        -------
        list[Agent]
        """
        agents: list[Agent] = []
        used_names: set[str] = set()
        used_occupations: set[str] = set()
        used_thinking: set[ThinkingStyle] = set()

        for i in range(count):
            # Spread temperatures evenly across the range
            if count > 1:
                frac = i / (count - 1)
                temp = temperature_range[0] + frac * (
                    temperature_range[1] - temperature_range[0]
                )
            else:
                temp = sum(temperature_range) / 2

            # Round to 2 decimal places
            temp = round(temp, 2)

            # Pick unique attributes when possible
            name = None
            occupation = None
            thinking = None

            if ensure_diversity:
                available_names = [n for n in _FIRST_NAMES if n not in used_names]
                if available_names:
                    name = self._rng.choice(available_names)
                    used_names.add(name)

                available_occ = [o for o in _OCCUPATIONS if o not in used_occupations]
                if available_occ:
                    occupation = self._rng.choice(available_occ)
                    used_occupations.add(occupation)

                available_think = [t for t in ThinkingStyle if t not in used_thinking]
                if available_think:
                    thinking = self._rng.choice(available_think)
                    used_thinking.add(thinking)

            agent = self.create_agent(
                name=name,
                occupation=occupation,
                temperature=temp,
                thinking_style=thinking,
            )
            agents.append(agent)

        return agents

    def create_custom_group(self, specs: list[dict]) -> list[Agent]:
        """
        Create agents from a list of specification dictionaries.

        Each dict may contain any of the following keys::

            name, age, occupation, temperature, thinking_style,
            education, background, personality_traits, values,
            communication_style

        Missing keys are filled with random values.

        Example::

            specs = [
                {"name": "Alice", "temperature": 0.3,
                 "thinking_style": ThinkingStyle.ANALYTICAL},
                {"name": "Bob", "temperature": 1.1,
                 "thinking_style": ThinkingStyle.CREATIVE},
            ]
        """
        agents: list[Agent] = []
        for spec in specs:
            agent = self.create_agent(**spec)
            agents.append(agent)
        return agents
