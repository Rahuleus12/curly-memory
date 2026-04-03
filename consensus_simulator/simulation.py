"""
simulation.py - Core simulation engine for multi-agent consensus discussions.

This module orchestrates the entire simulation lifecycle:
  1. Initialising agents with distinct personas and temperature settings
  2. Running multi-round discussions where agents respond to a question
     and to each other
  3. Detecting consensus (or the lack thereof)
  4. Generating a final consensus report via an impartial LLM facilitator
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from consensus_simulator.agents import Agent, AgentFactory
from consensus_simulator.prompts import (
    CONSENSUS_PROMPT,
    INITIAL_RESPONSE_PROMPT,
    MODERATOR_SUMMARY_PROMPT,
    ROUND_RESPONSE_PROMPT,
    build_initial_prompt,
    build_round_prompt,
    build_system_prompt,
    get_random_personas,
)
from openai import OpenAI

# ---------------------------------------------------------------------------
# Data structures for a simulation run
# ---------------------------------------------------------------------------


@dataclass
class RoundResponse:
    """A single agent's response within one round."""

    agent_name: str
    round_number: int
    temperature: float
    thinking_style: str
    text: str
    stance: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class RoundResult:
    """Aggregated results for one discussion round."""

    round_number: int
    responses: list[RoundResponse] = field(default_factory=list)
    moderator_summary: Optional[str] = None
    consensus_reached: bool = False
    stance_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Complete results of a simulation run."""

    question: str
    context: str = ""
    category: str = "general"
    rounds: list[RoundResult] = field(default_factory=list)
    final_consensus: Optional[dict[str, Any]] = None
    total_rounds: int = 0
    consensus_reached: bool = False
    transcript: str = ""
    agent_profiles: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "question": self.question,
            "context": self.context,
            "category": self.category,
            "total_rounds": self.total_rounds,
            "consensus_reached": self.consensus_reached,
            "final_consensus": self.final_consensus,
            "transcript": self.transcript,
            "agent_profiles": self.agent_profiles,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "moderator_summary": r.moderator_summary,
                    "consensus_reached": r.consensus_reached,
                    "stance_distribution": r.stance_distribution,
                    "responses": [
                        {
                            "agent_name": resp.agent_name,
                            "temperature": resp.temperature,
                            "thinking_style": resp.thinking_style,
                            "text": resp.text,
                            "stance": resp.stance,
                            "timestamp": resp.timestamp,
                        }
                        for resp in r.responses
                    ],
                }
                for r in self.rounds
            ],
        }


# ---------------------------------------------------------------------------
# Stance extraction helpers
# ---------------------------------------------------------------------------

_STANCE_RE = re.compile(r"<stance>(.*?)</stance>", re.IGNORECASE | re.DOTALL)

_AGREE_KEYWORDS = {
    "strongly support",
    "strongly favor",
    "strongly agree",
    "i support",
    "i favor",
    "i agree",
    "i'm in favor",
    "i'm for",
    "fully support",
    "definitely support",
    "absolutely support",
}

_DISAGREE_KEYWORDS = {
    "strongly oppose",
    "strongly against",
    "strongly disagree",
    "i oppose",
    "i'm against",
    "i disagree",
    "cannot support",
    "do not support",
    "firmly against",
}

_NEUTRAL_KEYWORDS = {
    "neutral",
    "undecided",
    "on the fence",
    "mixed feelings",
    "ambivalent",
    "somewhat neutral",
}


def extract_stance(text: str) -> str:
    """
    Extract a normalised stance from an agent's response.

    Checks <stance> tags first, then falls back to keyword matching.
    Returns one of: ``"strongly for"``, ``"somewhat for"``, ``"neutral"``,
    ``"somewhat against"``, ``"strongly against"``, or ``"unclear"``.
    """
    # Try <stance> tags
    match = _STANCE_RE.search(text)
    if match:
        stance_raw = match.group(1).strip().lower()
        return _classify_stance_text(stance_raw)

    # Fall back to keyword search in the whole text
    lower = text.lower()
    for kw in _AGREE_KEYWORDS:
        if kw in lower:
            if "strongly" in lower or "firmly" in lower:
                return "strongly for"
            return "somewhat for"

    for kw in _DISAGREE_KEYWORDS:
        if kw in lower:
            if "strongly" in lower or "firmly" in lower:
                return "strongly against"
            return "somewhat against"

    for kw in _NEUTRAL_KEYWORDS:
        if kw in lower:
            return "neutral"

    return "unclear"


def _classify_stance_text(text: str) -> str:
    """Map a free-text stance to one of our canonical buckets."""
    text = text.lower().strip().rstrip(".")

    if any(
        w in text
        for w in (
            "strongly support",
            "strongly favor",
            "strongly agree",
            "fully support",
        )
    ):
        return "strongly for"
    if any(w in text for w in ("support", "favor", "agree", "for", "in favor", "pro")):
        return "somewhat for"
    if any(
        w in text
        for w in (
            "strongly oppose",
            "strongly against",
            "strongly disagree",
            "firmly against",
        )
    ):
        return "strongly against"
    if any(w in text for w in ("oppose", "against", "disagree", "anti", "reject")):
        return "somewhat against"
    if any(
        w in text
        for w in ("neutral", "undecided", "mixed", "ambivalent", "on the fence")
    ):
        return "neutral"
    if any(
        w in text
        for w in ("lean toward", "somewhat support", "mildly support", "tend to agree")
    ):
        return "somewhat for"
    if any(
        w in text
        for w in (
            "lean against",
            "somewhat oppose",
            "mildly oppose",
            "tend to disagree",
        )
    ):
        return "somewhat against"

    return "unclear"


# ---------------------------------------------------------------------------
# Consensus detection
# ---------------------------------------------------------------------------


def compute_stance_distribution(responses: list[RoundResponse]) -> dict[str, int]:
    """Count how many agents fall into each stance bucket."""
    dist: dict[str, int] = {}
    for resp in responses:
        stance = resp.stance or "unclear"
        dist[stance] = dist.get(stance, 0) + 1
    return dist


def check_consensus(
    responses: list[RoundResponse],
    threshold: float = 0.7,
) -> tuple[bool, str]:
    """
    Determine whether the group has reached consensus.

    Parameters
    ----------
    responses : list[RoundResponse]
        The latest round's responses.
    threshold : float
        Fraction of agents that must share a position.

    Returns
    -------
    tuple[bool, str]
        ``(reached, dominant_stance)``
    """
    if not responses:
        return False, "none"

    dist = compute_stance_distribution(responses)
    total = len(responses)

    # Group "strongly for" + "somewhat for" and similarly for "against"
    grouped: dict[str, int] = {"for": 0, "against": 0, "neutral": 0, "unclear": 0}
    for stance, count in dist.items():
        if "for" in stance:
            grouped["for"] += count
        elif "against" in stance:
            grouped["against"] += count
        elif stance == "neutral":
            grouped["neutral"] += count
        else:
            grouped["unclear"] += count

    for position, count in grouped.items():
        if count / total >= threshold:
            return True, position

    return False, "none"


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def build_transcript(rounds: list[RoundResult]) -> str:
    """Render all rounds into a readable transcript string."""
    parts: list[str] = []
    for rnd in rounds:
        parts.append(f"\n{'=' * 60}")
        parts.append(f"  ROUND {rnd.round_number}")
        parts.append(f"{'=' * 60}\n")
        for resp in rnd.responses:
            parts.append(
                f"[{resp.agent_name}] (temp={resp.temperature}, style={resp.thinking_style})"
            )
            parts.append(resp.text)
            if resp.stance:
                parts.append(f"  → Stance: {resp.stance}")
            parts.append("")
        if rnd.moderator_summary:
            parts.append(f"--- Moderator Summary ---")
            parts.append(rnd.moderator_summary)
            parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """
    High-level engine that drives a complete consensus simulation.

    Typical usage::

        engine = SimulationEngine(client=my_openai_client, model="gpt-4")
        result = engine.run(
            question="Should AI be regulated?",
            agents=my_agents,
        )
        engine.save_result(result, "output/my_sim.json")
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4",
        max_rounds: int = 5,
        consensus_threshold: float = 0.7,
        parallel: bool = True,
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.parallel = parallel
        self.verbose = verbose

    # --- logging helper ----------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            print(f"[{timestamp}] {msg}")

    # --- single-agent response via prompts.py templates --------------------

    def _get_agent_response_with_template(
        self,
        agent: Agent,
        question: str,
        round_number: int,
        discussion_history: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate an agent's response using the prompts.py template system.

        This creates a system prompt from the agent's profile and uses
        the structured discussion templates.
        """
        # Build a persona dict compatible with prompts.py
        persona = {
            "name": agent.profile.name,
            "age": agent.profile.age,
            "occupation": agent.profile.occupation,
            "background": agent.profile.background,
            "personality_traits": agent.profile.personality_traits,
            "communication_style": agent.profile.communication_style,
            "biases": agent.profile.values[:3] if agent.profile.values else [],
            "expertise_areas": agent.profile.values[:2] if agent.profile.values else [],
        }

        system_prompt = build_system_prompt(persona)

        if round_number == 1:
            user_prompt = build_initial_prompt(question, persona)
        else:
            user_prompt = build_round_prompt(
                question, persona, round_number, discussion_history
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=agent.config.temperature,
                max_tokens=agent.config.max_tokens,
                top_p=agent.config.top_p,
                frequency_penalty=agent.config.frequency_penalty,
                presence_penalty=agent.config.presence_penalty,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"[Error generating response: {exc}]"

    # --- moderator summary -------------------------------------------------

    def _generate_moderator_summary(
        self,
        question: str,
        round_number: int,
        responses: list[RoundResponse],
    ) -> str:
        """Ask the LLM to summarise a round as an impartial moderator."""
        round_text = "\n\n".join(
            f"**{r.agent_name}** (temp={r.temperature}):\n{r.text}" for r in responses
        )

        prompt = MODERATOR_SUMMARY_PROMPT.format(
            topic=question,
            round_number=round_number,
            round_responses=round_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an impartial discussion moderator. Summarise concisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return f"[Could not generate moderator summary: {exc}]"

    # --- consensus analysis ------------------------------------------------

    def _generate_consensus_analysis(
        self,
        question: str,
        transcript: str,
    ) -> dict[str, Any]:
        """Use the consensus prompt to produce a final analysis."""
        prompt = CONSENSUS_PROMPT.format(
            topic=question,
            transcript=transcript,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an impartial facilitator analysing a group discussion. "
                            "Respond ONLY with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```\s*$", "", raw)

            return json.loads(raw)

        except json.JSONDecodeError:
            return {
                "consensus_level": "unknown",
                "areas_of_agreement": [],
                "areas_of_disagreement": [],
                "consensus_statement": raw
                if "raw" in dir()
                else "Could not parse analysis.",
                "key_perspectives": {},
                "raw_response": raw if "raw" in dir() else "",
            }
        except Exception as exc:
            return {
                "consensus_level": "error",
                "error": str(exc),
                "consensus_statement": "Analysis generation failed.",
            }

    # --- discussion history helper -----------------------------------------

    @staticmethod
    def _format_history(rounds: list[RoundResult], up_to_round: int = -1) -> str:
        """Build a readable discussion history from previous rounds."""
        target_rounds = rounds if up_to_round == -1 else rounds[:up_to_round]
        parts: list[str] = []
        for rnd in target_rounds:
            parts.append(f"--- Round {rnd.round_number} ---")
            for resp in rnd.responses:
                parts.append(f"{resp.agent_name}: {resp.text}")
            parts.append("")
        return "\n".join(parts)

    # --- main simulation loop ----------------------------------------------

    def run(
        self,
        question: str,
        agents: list[Agent],
        context: Optional[str] = None,
        category: str = "general",
        use_template_prompts: bool = True,
    ) -> SimulationResult:
        """
        Execute a full multi-round consensus simulation.

        Parameters
        ----------
        question : str
            The question or topic to discuss.
        agents : list[Agent]
            The simulated participants.
        context : str, optional
            Additional background information.
        category : str
            Topic category label.
        use_template_prompts : bool
            If ``True`` (default), uses the structured prompt templates from
            ``prompts.py``.  Otherwise delegates to ``Agent.respond()``.

        Returns
        -------
        SimulationResult
        """
        started_at = datetime.utcnow().isoformat()
        self._log(f'Starting simulation: "{question[:80]}..."')
        self._log(
            f"Agents: {len(agents)} | Max rounds: {self.max_rounds} | "
            f"Threshold: {self.consensus_threshold}"
        )

        result = SimulationResult(
            question=question,
            context=context or "",
            category=category,
            started_at=started_at,
            agent_profiles=[
                {
                    "name": a.profile.name,
                    "age": a.profile.age,
                    "occupation": a.profile.occupation,
                    "education": a.profile.education.value,
                    "thinking_style": a.profile.thinking_style.value,
                    "temperature": a.config.temperature,
                    "personality_traits": a.profile.personality_traits,
                    "values": a.profile.values,
                }
                for a in agents
            ],
        )

        consensus_reached = False

        for round_num in range(1, self.max_rounds + 1):
            self._log(f"--- Round {round_num}/{self.max_rounds} ---")
            round_result = RoundResult(round_number=round_num)

            # Build discussion history
            history = self._format_history(result.rounds) if result.rounds else ""

            # Gather responses — optionally in parallel
            if self.parallel and len(agents) > 1:
                round_responses = self._gather_parallel(
                    agents, question, round_num, history, context, use_template_prompts
                )
            else:
                round_responses = self._gather_sequential(
                    agents, question, round_num, history, context, use_template_prompts
                )

            round_result.responses = round_responses

            # Extract stances
            for resp in round_responses:
                resp.stance = extract_stance(resp.text)
                self._log(f"  {resp.agent_name}: stance={resp.stance}")

            # Compute distribution
            round_result.stance_distribution = compute_stance_distribution(
                round_responses
            )
            self._log(f"  Distribution: {round_result.stance_distribution}")

            # Generate moderator summary
            round_result.moderator_summary = self._generate_moderator_summary(
                question, round_num, round_responses
            )
            self._log(f"  Summary: {round_result.moderator_summary[:120]}...")

            # Check consensus
            reached, dominant = check_consensus(
                round_responses, self.consensus_threshold
            )
            round_result.consensus_reached = reached

            result.rounds.append(round_result)

            if reached:
                consensus_reached = True
                self._log(f"  ✓ Consensus reached! Dominant position: {dominant}")
                # Run one final round where agents confirm
                if round_num < self.max_rounds:
                    self._log("  Running confirmation round...")
                else:
                    break
            elif round_num == self.max_rounds:
                self._log("  Max rounds reached without consensus.")
            else:
                self._log("  No consensus yet, continuing...")

        # Build transcript
        result.transcript = build_transcript(result.rounds)
        result.total_rounds = len(result.rounds)
        result.consensus_reached = consensus_reached

        # Generate final consensus analysis
        self._log("Generating final consensus analysis...")
        result.final_consensus = self._generate_consensus_analysis(
            question, result.transcript
        )

        result.finished_at = datetime.utcnow().isoformat()
        self._log("Simulation complete.")
        return result

    # --- parallel / sequential gathering -----------------------------------

    def _gather_parallel(
        self,
        agents: list[Agent],
        question: str,
        round_number: int,
        history: str,
        context: Optional[str],
        use_template_prompts: bool,
    ) -> list[RoundResponse]:
        """Gather agent responses using threads for parallelism."""

        def _call(agent: Agent) -> RoundResponse:
            if use_template_prompts:
                text = self._get_agent_response_with_template(
                    agent, question, round_number, history, context
                )
            else:
                previous = (
                    [
                        {"name": r.agent_name, "text": r.text}
                        for rnd in result_rounds
                        for r in rnd.responses
                    ]
                    if "result_rounds" in dir()
                    else []
                )
                text = agent.respond(
                    question=question,
                    previous_responses=previous,
                    context=context,
                    round_number=round_number,
                )

            return RoundResponse(
                agent_name=agent.profile.name,
                round_number=round_number,
                temperature=agent.config.temperature,
                thinking_style=agent.profile.thinking_style.value,
                text=text,
            )

        responses: list[RoundResponse] = []
        with ThreadPoolExecutor(max_workers=min(len(agents), 10)) as executor:
            futures = {executor.submit(_call, a): a for a in agents}
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    resp = future.result(timeout=120)
                    responses.append(resp)
                except Exception as exc:
                    self._log(f"  Error from {agent.profile.name}: {exc}")
                    responses.append(
                        RoundResponse(
                            agent_name=agent.profile.name,
                            round_number=round_number,
                            temperature=agent.config.temperature,
                            thinking_style=agent.profile.thinking_style.value,
                            text=f"[Error: {exc}]",
                        )
                    )

        # Sort by agent name for deterministic ordering
        responses.sort(key=lambda r: r.agent_name)
        return responses

    def _gather_sequential(
        self,
        agents: list[Agent],
        question: str,
        round_number: int,
        history: str,
        context: Optional[str],
        use_template_prompts: bool,
    ) -> list[RoundResponse]:
        """Gather agent responses one at a time."""

        responses: list[RoundResponse] = []
        for agent in agents:
            self._log(f"  Querying {agent.profile.name}...")

            if use_template_prompts:
                text = self._get_agent_response_with_template(
                    agent, question, round_number, history, context
                )
            else:
                # Build previous responses from earlier rounds
                previous: list[dict[str, str]] = []
                for resp in responses:
                    previous.append({"name": resp.agent_name, "text": resp.text})
                text = agent.respond(
                    question=question,
                    previous_responses=previous or None,
                    context=context,
                    round_number=round_number,
                )

            responses.append(
                RoundResponse(
                    agent_name=agent.profile.name,
                    round_number=round_number,
                    temperature=agent.config.temperature,
                    thinking_style=agent.profile.thinking_style.value,
                    text=text,
                )
            )

        return responses

    # --- persistence -------------------------------------------------------

    @staticmethod
    def save_result(result: SimulationResult, path: str | Path) -> None:
        """Save a simulation result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_result(path: str | Path) -> SimulationResult:
        """Load a simulation result from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = SimulationResult(
            question=data["question"],
            context=data.get("context", ""),
            category=data.get("category", "general"),
            total_rounds=data.get("total_rounds", 0),
            consensus_reached=data.get("consensus_reached", False),
            final_consensus=data.get("final_consensus"),
            transcript=data.get("transcript", ""),
            agent_profiles=data.get("agent_profiles", []),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
        )

        for rnd_data in data.get("rounds", []):
            rnd = RoundResult(
                round_number=rnd_data["round_number"],
                moderator_summary=rnd_data.get("moderator_summary"),
                consensus_reached=rnd_data.get("consensus_reached", False),
                stance_distribution=rnd_data.get("stance_distribution", {}),
            )
            for resp_data in rnd_data.get("responses", []):
                rnd.responses.append(
                    RoundResponse(
                        agent_name=resp_data["agent_name"],
                        round_number=resp_data["round_number"],
                        temperature=resp_data.get("temperature", 0.7),
                        thinking_style=resp_data.get("thinking_style", "unknown"),
                        text=resp_data["text"],
                        stance=resp_data.get("stance"),
                        timestamp=resp_data.get("timestamp", ""),
                    )
                )
            result.rounds.append(rnd)

        return result


# ---------------------------------------------------------------------------
# Convenience: run simulation from simple parameters
# ---------------------------------------------------------------------------


def quick_simulate(
    question: str,
    num_agents: int = 5,
    temperature_range: tuple[float, float] = (0.3, 1.2),
    max_rounds: int = 4,
    consensus_threshold: float = 0.7,
    model: str = "gpt-4",
    api_key: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    seed: Optional[int] = None,
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> SimulationResult:
    """
    One-call convenience function for running a simulation.

    Creates a random diverse group of agents and runs the full discussion.

    Parameters
    ----------
    question : str
        The topic to discuss.
    num_agents : int
        How many simulated people to create.
    temperature_range : tuple
        ``(min_temp, max_temp)`` spread across agents.
    max_rounds : int
        Maximum discussion rounds.
    consensus_threshold : float
        Fraction needed for consensus (0.5–1.0).
    model : str
        LLM model identifier.
    api_key : str, optional
        OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
    api_base : str
        API base URL.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print progress to console.
    output_path : str, optional
        If provided, save results to this path.

    Returns
    -------
    SimulationResult
    """
    import os

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "No API key provided. Set api_key parameter or OPENAI_API_KEY env var."
        )

    client = OpenAI(api_key=key, base_url=api_base)

    factory = AgentFactory(client=client, model=model, seed=seed)
    agents = factory.create_group(
        count=num_agents,
        temperature_range=temperature_range,
        ensure_diversity=True,
    )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  CONSENSUS SIMULATION")
        print(f"{'=' * 60}")
        print(f"  Question: {question}")
        print(
            f"  Agents: {num_agents} | Rounds: {max_rounds} | Threshold: {consensus_threshold}"
        )
        print(f"  Temperature range: {temperature_range}")
        print(f"{'=' * 60}\n")
        for a in agents:
            print(f"  • {a}")

    engine = SimulationEngine(
        client=client,
        model=model,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
        verbose=verbose,
    )

    result = engine.run(question=question, agents=agents)

    if output_path:
        engine.save_result(result, output_path)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return result
