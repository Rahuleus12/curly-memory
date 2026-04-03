"""
Utility helpers for the Consensus Simulator.

Provides functions for:
- Formatting console output with rich panels and tables
- Saving simulation results to JSON and Markdown files
- Parsing and extracting stances from agent responses
- Generating timestamps and safe filenames
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def timestamp_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a filename-safe timestamp string."""
    return utc_now().strftime(fmt)


def safe_filename(name: str) -> str:
    """
    Convert an arbitrary string into a safe filename component.

    Non-alphanumeric characters are replaced with underscores and
    consecutive underscores are collapsed.
    """
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_").lower()


# ---------------------------------------------------------------------------
# Stance extraction
# ---------------------------------------------------------------------------


def extract_stance_tag(text: str) -> Optional[str]:
    """
    Extract the content of a ``<stance>...</stance>`` tag from *text*.

    Returns the stripped tag content, or ``None`` if no tag is found.
    """
    match = re.search(r"<stance>\s*(.+?)\s*</stance>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def classify_stance(text: str) -> str:
    """
    Classify an agent's stance as one of: strongly_for, somewhat_for,
    neutral, somewhat_against, strongly_against, or unclear.

    The function first looks for a ``<stance>`` tag.  If none is found
    it falls back to keyword heuristics on the full text.
    """
    stance_text = extract_stance_tag(text) or text

    lower = stance_text.lower()

    # Strong signals
    strongly_for_kw = [
        "strongly support",
        "strongly favor",
        "strongly agree",
        "strongly for",
        "fully support",
        "enthusiastically",
        "wholeheartedly",
        "absolutely support",
        "definitely support",
        "strongly in favor",
    ]
    strongly_against_kw = [
        "strongly oppose",
        "strongly against",
        "strongly disagree",
        "firmly against",
        "categorically oppose",
        "absolutely oppose",
        "strongly object",
    ]

    for kw in strongly_for_kw:
        if kw in lower:
            return "strongly_for"
    for kw in strongly_against_kw:
        if kw in lower:
            return "strongly_against"

    # Moderate signals
    somewhat_for_kw = [
        "i support",
        "i agree",
        "i'm in favor",
        "i favor",
        "i'm for",
        "generally support",
        "lean toward supporting",
        "tend to agree",
        "mostly agree",
        "i'm inclined to support",
        "i believe we should",
    ]
    somewhat_against_kw = [
        "i oppose",
        "i disagree",
        "i'm against",
        "i'm not in favor",
        "lean toward opposing",
        "tend to disagree",
        "mostly disagree",
        "cannot support",
        "do not support",
    ]

    for kw in somewhat_for_kw:
        if kw in lower:
            return "somewhat_for"
    for kw in somewhat_against_kw:
        if kw in lower:
            return "somewhat_against"

    # Neutral signals
    neutral_kw = [
        "neutral",
        "undecided",
        "on the fence",
        "mixed feelings",
        "ambivalent",
        "can see both sides",
        "neither for nor against",
        "no strong opinion",
        "i'm torn",
        "balance of",
        "need more information",
        "need more data",
    ]
    for kw in neutral_kw:
        if kw in lower:
            return "neutral"

    return "unclear"


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap *text* to *width* characters per line."""
    return "\n".join(textwrap.wrap(text, width=width))


def indent_text(text: str, indent: int = 2) -> str:
    """Indent every line of *text* by *indent* spaces."""
    prefix = " " * indent
    return textwrap.indent(text, prefix)


def format_agent_response(
    name: str, response: str, stance: Optional[str] = None
) -> str:
    """
    Format a single agent response for display.

    Parameters
    ----------
    name : str
        Agent name.
    response : str
        The response text.
    stance : str, optional
        Classified stance label (e.g. "strongly_for").

    Returns
    -------
    str
        Formatted string ready for console or file output.
    """
    lines = [f"┌─ {name}"]
    if stance:
        lines[0] += f"  [{stance}]"
    lines.append("│")

    for paragraph in response.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=76)
        for line in wrapped:
            lines.append(f"│  {line}")
        lines.append("│")

    lines.append("└" + "─" * 77)
    return "\n".join(lines)


def format_round_header(round_number: int, max_rounds: int) -> str:
    """Return a visual header for a discussion round."""
    width = 60
    title = f"  ROUND {round_number} / {max_rounds}  "
    pad = width - len(title)
    left = pad // 2
    right = pad - left
    line = "═" * left + title + "═" * right
    return f"\n╔{line}╗\n║{' ' * width}║\n╚{'═' * width}╝"


def format_divider(char: str = "─", width: int = 60) -> str:
    """Return a horizontal divider line."""
    return char * width


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it doesn't exist; return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, indent: int = 2) -> Path:
    """
    Save *data* as JSON to *path*.

    Returns the actual path written (parent directories are created).
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    return path


def load_json(path: Path) -> Any:
    """Load and return JSON data from *path*."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_markdown(content: str, path: Path) -> Path:
    """Write *content* as a Markdown file to *path*."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_transcript_markdown(
    question: str,
    rounds: list[dict],
    summary: Optional[dict] = None,
) -> str:
    """
    Build a full Markdown transcript of a simulation.

    Parameters
    ----------
    question : str
        The discussion question.
    rounds : list[dict]
        Each dict has ``"round_number"`` and ``"responses"`` (list of
        ``{"name", "text", "stance"}``).
    summary : dict, optional
        Final summary with keys like ``consensus_level``,
        ``consensus_statement``, etc.

    Returns
    -------
    str
        Complete Markdown string.
    """
    parts: list[str] = []

    parts.append("# Consensus Simulation Transcript\n")
    parts.append(f"**Generated:** {utc_now().isoformat()}\n")
    parts.append(f"## Question\n\n{question}\n")

    # Agent roster from round 1
    if rounds and rounds[0].get("responses"):
        parts.append("## Participants\n")
        for resp in rounds[0]["responses"]:
            parts.append(f"- **{resp['name']}**")
        parts.append("")

    # Rounds
    for rnd in rounds:
        rn = rnd.get("round_number", "?")
        parts.append(f"## Round {rn}\n")
        for resp in rnd.get("responses", []):
            name = resp["name"]
            text = resp["text"]
            stance = resp.get("stance", "")
            parts.append(f"### {name}\n")
            if stance:
                parts.append(f"*Stance: {stance}*\n")
            parts.append(f"{text}\n")
            parts.append("---\n")

    # Summary
    if summary:
        parts.append("## Summary\n")
        cl = summary.get("consensus_level", "N/A")
        parts.append(f"**Consensus Level:** {cl} / 5\n")

        agreement = summary.get("areas_of_agreement", [])
        if agreement:
            parts.append("### Areas of Agreement\n")
            if isinstance(agreement, list):
                for item in agreement:
                    parts.append(f"- {item}")
            else:
                parts.append(str(agreement))
            parts.append("")

        disagreement = summary.get("areas_of_disagreement", [])
        if disagreement:
            parts.append("### Areas of Disagreement\n")
            if isinstance(disagreement, list):
                for item in disagreement:
                    parts.append(f"- {item}")
            else:
                parts.append(str(disagreement))
            parts.append("")

        cs = summary.get("consensus_statement", "")
        if cs:
            parts.append("### Consensus Statement\n")
            parts.append(f"{cs}\n")

        kp = summary.get("key_perspectives", {})
        if kp:
            parts.append("### Key Perspectives\n")
            if isinstance(kp, dict):
                for pname, pval in kp.items():
                    parts.append(f"- **{pname}:** {pval}")
            elif isinstance(kp, list):
                for item in kp:
                    parts.append(f"- {item}")
            parts.append("")

    return "\n".join(parts)


def build_summary_markdown(
    question: str,
    total_rounds: int,
    agent_summaries: list[dict],
    consensus_info: Optional[dict] = None,
) -> str:
    """
    Build a shorter summary report (not the full transcript).

    Parameters
    ----------
    question : str
        The discussion question.
    total_rounds : int
        Number of rounds completed.
    agent_summaries : list[dict]
        Each dict has ``name``, ``initial_stance``, ``final_stance``,
        ``stance_changed``.
    consensus_info : dict, optional
        Aggregated consensus data.

    Returns
    -------
    str
        Markdown summary string.
    """
    parts: list[str] = []

    parts.append("# Consensus Simulation Summary\n")
    parts.append(f"**Generated:** {utc_now().isoformat()}\n")
    parts.append(f"**Question:** {question}\n")
    parts.append(f"**Rounds:** {total_rounds}\n")

    # Stance table
    parts.append("## Agent Stances\n")
    parts.append("| Agent | Initial Stance | Final Stance | Changed? |")
    parts.append("|-------|---------------|-------------|----------|")
    for a in agent_summaries:
        changed = "✓" if a.get("stance_changed") else "—"
        parts.append(
            f"| {a['name']} | {a.get('initial_stance', '?')} "
            f"| {a.get('final_stance', '?')} | {changed} |"
        )
    parts.append("")

    # Stance distribution
    if agent_summaries:
        parts.append("## Final Stance Distribution\n")
        stance_counts: dict[str, int] = {}
        for a in agent_summaries:
            fs = a.get("final_stance", "unclear")
            stance_counts[fs] = stance_counts.get(fs, 0) + 1
        for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count * 3)
            parts.append(f"- **{stance}:** {count}  {bar}")
        parts.append("")

    # Consensus info
    if consensus_info:
        parts.append("## Consensus Result\n")
        cl = consensus_info.get("consensus_level", "N/A")
        parts.append(f"- **Consensus Level:** {cl} / 5\n")
        cs = consensus_info.get("consensus_statement", "")
        if cs:
            parts.append(f"- **Consensus Statement:** {cs}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------


def calculate_consensus_level(stances: list[str]) -> float:
    """
    Calculate a simple consensus score based on stance distribution.

    The score is the fraction of agents that hold the most common stance.

    Parameters
    ----------
    stances : list[str]
        Classified stances for all agents.

    Returns
    -------
    float
        Consensus score between 0.0 and 1.0.
    """
    if not stances:
        return 0.0

    counts: dict[str, int] = {}
    for s in stances:
        counts[s] = counts.get(s, 0) + 1

    max_count = max(counts.values())
    return max_count / len(stances)


def group_stances(stances: list[str]) -> dict[str, int]:
    """Return a dict mapping stance label → count."""
    counts: dict[str, int] = {}
    for s in stances:
        counts[s] = counts.get(s, 0) + 1
    return counts


def stance_to_numeric(stance: str) -> float:
    """
    Map a stance label to a numeric value.

    strongly_for → 2.0
    somewhat_for → 1.0
    neutral      → 0.0
    somewhat_against → -1.0
    strongly_against → -2.0
    unclear      → 0.0
    """
    mapping = {
        "strongly_for": 2.0,
        "somewhat_for": 1.0,
        "neutral": 0.0,
        "somewhat_against": -1.0,
        "strongly_against": -2.0,
        "unclear": 0.0,
    }
    return mapping.get(stance, 0.0)


def calculate_average_sentiment(stances: list[str]) -> float:
    """
    Calculate the average numeric sentiment across all agents.

    Positive values indicate overall support; negative values indicate
    overall opposition.
    """
    if not stances:
        return 0.0
    return sum(stance_to_numeric(s) for s in stances) / len(stances)
