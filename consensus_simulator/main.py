"""
main.py - CLI entry point for the Consensus Simulator.

Provides a rich terminal interface for running multi-agent consensus
simulations with varied temperature settings and persona prompts.

Usage examples:
    # Quick simulation with defaults
    python -m consensus_simulator "Should AI be regulated?"

    # Custom configuration
    python -m consensus_simulator "Is UBI a good idea?" --agents 6 --rounds 4 --model gpt-4

    # Use a config file
    python -m consensus_simulator --config simulation_config.json

    # List available personas
    python -m consensus_simulator --list-personas
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from consensus_simulator.agents import (
    Agent,
    AgentFactory,
    EducationLevel,
    ThinkingStyle,
)
from consensus_simulator.prompts import (
    PERSONA_TEMPLATES,
    build_system_prompt,
    get_all_persona_names,
    get_persona_by_index,
    get_personas_by_indices,
    get_random_personas,
)
from consensus_simulator.simulation import (
    RoundResponse,
    SimulationEngine,
    SimulationResult,
    check_consensus,
    extract_stance,
    quick_simulate,
)
from consensus_simulator.utils import (
    build_summary_markdown,
    build_transcript_markdown,
    calculate_average_sentiment,
    calculate_consensus_level,
    group_stances,
    safe_filename,
    save_json,
    save_markdown,
    stance_to_numeric,
    timestamp_str,
)

# ---------------------------------------------------------------------------
# Rich console & theme
# ---------------------------------------------------------------------------

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
        "success": "bold green",
        "agent": "yellow",
        "round": "bold blue",
        "consensus": "bold green",
        "stance_for": "green",
        "stance_against": "red",
        "stance_neutral": "yellow",
    }
)

console = Console(theme=custom_theme)
app = typer.Typer(
    name="consensus-simulator",
    help="Simulate a group of diverse people discussing topics to form consensus.",
    rich_markup_mode="rich",
)

# Load .env if present
load_dotenv()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_api_key(provided_key: Optional[str] = None) -> str:
    """Resolve the OpenAI API key from argument or environment."""
    key = provided_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        console.print(
            "[danger]Error:[/danger] No API key provided. "
            "Use --api-key or set the OPENAI_API_KEY environment variable."
        )
        raise typer.Exit(code=1)
    return key


def _get_api_base(provided_base: Optional[str] = None) -> str:
    """Resolve the API base URL."""
    return provided_base or os.environ.get(
        "OPENAI_API_BASE", "https://api.openai.com/v1"
    )


def _get_model(provided_model: Optional[str] = None) -> str:
    """Resolve the model name."""
    return provided_model or os.environ.get("OPENAI_MODEL", "gpt-4")


def _print_banner() -> None:
    """Print a stylish banner."""
    banner = Text()
    banner.append(
        "╔══════════════════════════════════════════════════════════╗\n",
        style="bold blue",
    )
    banner.append(
        "║                                                          ║\n",
        style="bold blue",
    )
    banner.append(
        "║          🤝  CONSENSUS SIMULATOR  🤝                     ║\n",
        style="bold cyan",
    )
    banner.append(
        "║                                                          ║\n",
        style="bold blue",
    )
    banner.append(
        "║   Simulate diverse perspectives reaching consensus       ║\n",
        style="dim cyan",
    )
    banner.append(
        "║                                                          ║\n",
        style="bold blue",
    )
    banner.append(
        "╚══════════════════════════════════════════════════════════╝\n",
        style="bold blue",
    )
    console.print(banner)


def _print_agents_table(agents: list[Agent]) -> None:
    """Display a rich table of agent profiles."""
    table = Table(title="🧑‍🤝‍🧑 Simulated Participants", show_lines=True)
    table.add_column("Name", style="agent", no_wrap=True)
    table.add_column("Age", justify="center")
    table.add_column("Occupation", max_width=22)
    table.add_column("Education", max_width=14)
    table.add_column("Thinking", max_width=14)
    table.add_column("Temperature", justify="center")
    table.add_column("Traits", max_width=30)

    for a in agents:
        traits = ", ".join(a.profile.personality_traits[:3])
        table.add_row(
            a.profile.name,
            str(a.profile.age),
            a.profile.occupation,
            a.profile.education.value.replace("_", " "),
            a.profile.thinking_style.value,
            f"{a.config.temperature:.2f}",
            traits,
        )

    console.print(table)
    console.print()


def _print_round_results(
    round_number: int,
    max_rounds: int,
    responses: list,
    stance_distribution: dict,
) -> None:
    """Print formatted results for a discussion round."""
    console.print()
    console.rule(f"[round]Round {round_number} / {max_rounds}[/round]")

    for resp in responses:
        stance_str = resp.stance or "unclear"
        if "for" in stance_str:
            stance_style = "stance_for"
        elif "against" in stance_str:
            stance_style = "stance_against"
        else:
            stance_style = "stance_neutral"

        panel_content = Text()
        panel_content.append(resp.text)
        panel_content.append("\n\n")
        panel_content.append(f"Stance: {stance_str}", style=stance_style)
        panel_content.append(
            f"  |  temp={resp.temperature:.2f}  |  style={resp.thinking_style}"
        )

        console.print(
            Panel(
                panel_content,
                title=f"[agent]{resp.agent_name}[/agent]",
                border_style="dim",
                padding=(1, 2),
            )
        )

    # Stance distribution bar
    console.print()
    dist_text = Text("Stance Distribution:  ")
    total = sum(stance_distribution.values())
    for stance, count in sorted(stance_distribution.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5)
        if "for" in stance:
            style = "stance_for"
        elif "against" in stance:
            style = "stance_against"
        else:
            style = "stance_neutral"
        dist_text.append(f"\n  {stance}: {count} ({pct:.0f}%) {bar}", style=style)
    console.print(dist_text)
    console.print()


def _print_consensus_result(result: SimulationResult) -> None:
    """Print the final consensus analysis."""
    console.print()
    console.rule("[consensus]📋 Final Consensus Analysis[/consensus]")

    if not result.final_consensus:
        console.print("[warning]No consensus analysis available.[/warning]")
        return

    consensus_data = result.final_consensus

    # Consensus level
    level = consensus_data.get("consensus_level", "N/A")
    level_str = str(level)
    if level_str.isdigit() or (len(level_str) >= 1 and level_str[0].isdigit()):
        try:
            numeric_level = float(level_str.split("/")[0].strip().split()[0])
            if numeric_level >= 4:
                level_style = "success"
            elif numeric_level >= 3:
                level_style = "warning"
            else:
                level_style = "danger"
        except (ValueError, IndexError):
            level_style = "info"
    else:
        level_style = "info"

    console.print(
        f"\n[bold]Consensus Level:[/bold] [{level_style}]{level}[/{level_style}]"
    )

    reached_text = "✅ Yes" if result.consensus_reached else "❌ No"
    console.print(f"[bold]Consensus Reached:[/bold] {reached_text}")
    console.print(f"[bold]Total Rounds:[/bold] {result.total_rounds}")

    # Areas of agreement
    agreement = consensus_data.get("areas_of_agreement", [])
    if agreement:
        console.print("\n[bold green]🤝 Areas of Agreement:[/bold green]")
        if isinstance(agreement, list):
            for item in agreement:
                console.print(f"  • {item}", style="green")
        else:
            console.print(f"  {agreement}", style="green")

    # Areas of disagreement
    disagreement = consensus_data.get("areas_of_disagreement", [])
    if disagreement:
        console.print("\n[bold red]⚡ Areas of Disagreement:[/bold red]")
        if isinstance(disagreement, list):
            for item in disagreement:
                console.print(f"  • {item}", style="red")
        else:
            console.print(f"  {disagreement}", style="red")

    # Consensus statement
    statement = consensus_data.get("consensus_statement", "")
    if statement:
        console.print(
            Panel(
                Markdown(statement),
                title="[bold]📝 Consensus Statement[/bold]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # Key perspectives
    perspectives = consensus_data.get("key_perspectives", {})
    if perspectives:
        console.print("\n[bold]🔑 Key Perspectives:[/bold]")
        if isinstance(perspectives, dict):
            for name, perspective in perspectives.items():
                console.print(f"  [agent]{name}[/agent]: {perspective}")
        elif isinstance(perspectives, list):
            for item in perspectives:
                console.print(f"  • {item}")

    console.print()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    question: str = typer.Argument(
        None,
        help="The question or topic for discussion.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a JSON configuration file.",
        exists=True,
    ),
    agents: int = typer.Option(
        5,
        "--agents",
        "-n",
        min=2,
        max=20,
        help="Number of simulated agents.",
    ),
    rounds: int = typer.Option(
        4,
        "--rounds",
        "-r",
        min=1,
        max=20,
        help="Maximum number of discussion rounds.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use (e.g. gpt-4, gpt-3.5-turbo).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="OpenAI API key.",
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="OpenAI-compatible API base URL.",
    ),
    temperature_min: float = typer.Option(
        0.3,
        "--temp-min",
        min=0.0,
        max=2.0,
        help="Minimum temperature for agents.",
    ),
    temperature_max: float = typer.Option(
        1.2,
        "--temp-max",
        min=0.0,
        max=2.0,
        help="Maximum temperature for agents.",
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-t",
        min=0.5,
        max=1.0,
        help="Consensus threshold (fraction of agents agreeing).",
    ),
    output_dir: Optional[Path] = typer.Option(
        Path("output"),
        "--output",
        "-o",
        help="Directory to save results.",
    ),
    save_json_flag: bool = typer.Option(
        True,
        "--save-json/--no-save-json",
        help="Save results as JSON.",
    ),
    save_markdown_flag: bool = typer.Option(
        True,
        "--save-markdown/--no-save-markdown",
        help="Save transcript as Markdown.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Enable verbose output.",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--sequential",
        help="Run agent queries in parallel.",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        help="Additional context for the discussion topic.",
    ),
    category: str = typer.Option(
        "general",
        "--category",
        help="Category label for the topic.",
    ),
) -> None:
    """
    Run a consensus simulation with diverse AI-simulated participants.

    Each agent has a unique persona (background, personality, values) and
    temperature setting, producing varied and realistic responses. The
    agents discuss the given question over multiple rounds and attempt
    to reach consensus.
    """
    _print_banner()

    # --- Load from config file if provided ---
    if config:
        console.print(f"[info]Loading configuration from {config}...[/info]")
        with open(config, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        question = cfg_data.get("question", question)
        agents = cfg_data.get("num_agents", agents)
        rounds = cfg_data.get("max_rounds", rounds)
        threshold = cfg_data.get("consensus_threshold", threshold)
        model = cfg_data.get("model", model)
        context = cfg_data.get("context", context)
        category = cfg_data.get("category", category)
        temperature_min = cfg_data.get("temperature_min", temperature_min)
        temperature_max = cfg_data.get("temperature_max", temperature_max)
        seed = cfg_data.get("seed", seed)

    # --- Validate ---
    if not question:
        console.print("[danger]Error:[/danger] Please provide a question or topic.")
        console.print(
            'Usage: [bold]python -m consensus_simulator "Your question here"[/bold]'
        )
        raise typer.Exit(code=1)

    resolved_key = _get_api_key(api_key)
    resolved_base = _get_api_base(api_base)
    resolved_model = _get_model(model)

    # --- Print simulation parameters ---
    console.print(
        Panel(
            f"[bold]Question:[/bold] {question}\n\n"
            f"[dim]Model:[/dim] {resolved_model}  |  "
            f"[dim]Agents:[/dim] {agents}  |  "
            f"[dim]Rounds:[/dim] {rounds}  |  "
            f"[dim]Threshold:[/dim] {threshold:.0%}\n"
            f"[dim]Temp range:[/dim] [{temperature_min:.2f}, {temperature_max:.2f}]  |  "
            f"[dim]Parallel:[/dim] {parallel}  |  "
            f"[dim]Seed:[/dim] {seed or 'random'}",
            title="_simulation Parameters",
            border_style="cyan",
        )
    )

    if context:
        console.print(f"[info]Context:[/info] {context}\n")

    # --- Create client and agents ---
    client = OpenAI(api_key=resolved_key, base_url=resolved_base)
    factory = AgentFactory(client=client, model=resolved_model, seed=seed)
    agent_list = factory.create_group(
        count=agents,
        temperature_range=(temperature_min, temperature_max),
        ensure_diversity=True,
    )

    # Display agent roster
    _print_agents_table(agent_list)

    # --- Run simulation ---
    engine = SimulationEngine(
        client=client,
        model=resolved_model,
        max_rounds=rounds,
        consensus_threshold=threshold,
        parallel=parallel,
        verbose=False,  # We handle output ourselves
    )

    start_time = time.time()

    with console.status("[bold cyan]Running simulation...[/bold cyan]", spinner="dots"):
        result = engine.run(
            question=question,
            agents=agent_list,
            context=context,
            category=category,
        )

    elapsed = time.time() - start_time

    # --- Display results round by round ---
    for rnd in result.rounds:
        _print_round_results(
            rnd.round_number,
            rounds,
            rnd.responses,
            rnd.stance_distribution,
        )
        if rnd.moderator_summary:
            console.print(
                Panel(
                    Markdown(rnd.moderator_summary),
                    title="[bold]📝 Moderator Summary[/bold]",
                    border_style="dim",
                    padding=(1, 2),
                )
            )

    # --- Display final consensus ---
    _print_consensus_result(result)

    # --- Timing ---
    console.print(f"[dim]Simulation completed in {elapsed:.1f} seconds.[/dim]")

    # --- Save outputs ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_q = safe_filename(question[:40])
    ts = timestamp_str()

    if save_json_flag:
        json_path = output_dir / f"sim_{safe_q}_{ts}.json"
        engine.save_result(result, json_path)
        console.print(f"[info]JSON results saved to:[/info] {json_path}")

    if save_markdown_flag:
        # Build and save transcript
        md_rounds = []
        for rnd in result.rounds:
            md_rounds.append(
                {
                    "round_number": rnd.round_number,
                    "responses": [
                        {
                            "name": r.agent_name,
                            "text": r.text,
                            "stance": r.stance,
                        }
                        for r in rnd.responses
                    ],
                }
            )

        transcript_md = build_transcript_markdown(
            question=question,
            rounds=md_rounds,
            summary=result.final_consensus,
        )
        md_path = output_dir / f"transcript_{safe_q}_{ts}.md"
        save_markdown(transcript_md, md_path)
        console.print(f"[info]Markdown transcript saved to:[/info] {md_path}")

        # Build and save summary
        agent_summaries = []
        if result.rounds:
            first_round = result.rounds[0]
            last_round = result.rounds[-1]
            for first_resp in first_round.responses:
                initial_stance = first_resp.stance or "unclear"
                final_stance = initial_stance
                for last_resp in last_round.responses:
                    if last_resp.agent_name == first_resp.agent_name:
                        final_stance = last_resp.stance or "unclear"
                        break
                agent_summaries.append(
                    {
                        "name": first_resp.agent_name,
                        "initial_stance": initial_stance,
                        "final_stance": final_stance,
                        "stance_changed": initial_stance != final_stance,
                    }
                )

        summary_md = build_summary_markdown(
            question=question,
            total_rounds=result.total_rounds,
            agent_summaries=agent_summaries,
            consensus_info=result.final_consensus,
        )
        summary_path = output_dir / f"summary_{safe_q}_{ts}.md"
        save_markdown(summary_md, summary_path)
        console.print(f"[info]Summary report saved to:[/info] {summary_path}")

    console.print()
    console.print("[success]✅ Simulation complete![/success]")


@app.command()
def quick(
    question: str = typer.Argument(
        ...,
        help="The question or topic for discussion.",
    ),
    agents: int = typer.Option(5, "--agents", "-n", min=2, max=15),
    rounds: int = typer.Option(3, "--rounds", "-r", min=1, max=10),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
    ),
) -> None:
    """
    Quick simulation with minimal configuration.

    Runs a fast consensus simulation with sensible defaults.
    """
    _print_banner()
    resolved_key = _get_api_key(api_key)
    resolved_model = _get_model(model)

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(
        f"[dim]Agents: {agents} | Rounds: {rounds} | Model: {resolved_model}[/dim]\n"
    )

    with console.status("[bold cyan]Simulating...[/bold cyan]", spinner="dots"):
        result = quick_simulate(
            question=question,
            num_agents=agents,
            max_rounds=rounds,
            model=resolved_model,
            api_key=resolved_key,
            verbose=False,
        )

    # Print brief results
    for rnd in result.rounds:
        console.print(f"\n[round]--- Round {rnd.round_number} ---[/round]")
        for resp in rnd.responses:
            console.print(
                f"  [agent]{resp.agent_name}[/agent]: "
                f"[dim]{resp.text[:100]}...[/dim] "
                f"[{resp.stance or 'unclear'}]"
            )

    _print_consensus_result(result)


@app.command(name="list-personas")
def list_personas() -> None:
    """List all available pre-defined persona templates."""
    _print_banner()
    console.print("\n[bold]Available Persona Templates:[/bold]\n")

    table = Table(show_lines=True)
    table.add_column("#", style="dim", justify="center", width=4)
    table.add_column("Name", style="agent", width=22)
    table.add_column("Age", justify="center", width=5)
    table.add_column("Occupation", width=32)
    table.add_column("Traits", width=40)

    for i, p in enumerate(PERSONA_TEMPLATES):
        traits = ", ".join(p["personality_traits"][:3])
        table.add_row(
            str(i),
            p["name"],
            str(p["age"]),
            p["occupation"],
            traits,
        )

    console.print(table)
    console.print(
        f"\n[dim]Use persona indices with --personas to select specific participants.[/dim]"
    )


@app.command(name="generate-config")
def generate_config(
    output: Path = typer.Option(
        Path("simulation_config.json"),
        "--output",
        "-o",
        help="Path to write the configuration file.",
    ),
) -> None:
    """Generate a sample configuration file."""
    sample_config = {
        "question": "Should artificial intelligence be regulated by governments?",
        "context": (
            "AI systems are becoming increasingly powerful and autonomous. "
            "Some argue regulation stifles innovation while others believe "
            "it's essential for public safety."
        ),
        "category": "technology",
        "num_agents": 5,
        "max_rounds": 4,
        "consensus_threshold": 0.7,
        "model": "gpt-4",
        "temperature_min": 0.3,
        "temperature_max": 1.2,
        "seed": None,
        "parallel": True,
    }

    output.write_text(json.dumps(sample_config, indent=2), encoding="utf-8")
    console.print(f"[success]Sample configuration written to:[/success] {output}")
    console.print("[dim]Edit the file and run with --config <path>[/dim]")


@app.command(name="analyze")
def analyze_results(
    path: Path = typer.Argument(
        ...,
        help="Path to a simulation result JSON file.",
        exists=True,
    ),
) -> None:
    """Analyze a previously saved simulation result."""
    _print_banner()
    console.print(f"[info]Loading results from {path}...[/info]\n")

    engine = SimulationEngine.__new__(SimulationEngine)
    result = SimulationEngine.load_result(path)

    console.print(
        Panel(
            f"[bold]Question:[/bold] {result.question}\n"
            f"[dim]Rounds: {result.total_rounds} | "
            f"Consensus: {'Yes' if result.consensus_reached else 'No'} | "
            f"Agents: {len(result.agent_profiles)}[/dim]",
            title="Simulation Results",
            border_style="cyan",
        )
    )

    # Replay round results
    for rnd in result.rounds:
        _print_round_results(
            rnd.round_number,
            result.total_rounds,
            rnd.responses,
            rnd.stance_distribution,
        )

    # Consensus analysis
    _print_consensus_result(result)

    # Stance evolution
    if len(result.rounds) > 1:
        console.rule("[bold]📊 Stance Evolution[/bold]")
        evolution_table = Table(title="How stances changed across rounds")
        evolution_table.add_column("Agent", style="agent")

        for rnd in result.rounds:
            evolution_table.add_column(f"Round {rnd.round_number}", justify="center")

        # Collect all agent names
        agent_names = []
        if result.rounds:
            for resp in result.rounds[0].responses:
                agent_names.append(resp.agent_name)

        for name in agent_names:
            row = [name]
            for rnd in result.rounds:
                for resp in rnd.responses:
                    if resp.agent_name == name:
                        stance = resp.stance or "unclear"
                        if "for" in stance:
                            row.append(f"[stance_for]{stance}[/stance_for]")
                        elif "against" in stance:
                            row.append(f"[stance_against]{stance}[/stance_against]")
                        else:
                            row.append(f"[stance_neutral]{stance}[/stance_neutral]")
                        break
            evolution_table.add_row(*row)

        console.print(evolution_table)
        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Package entry point."""
    app()


if __name__ == "__main__":
    main()
