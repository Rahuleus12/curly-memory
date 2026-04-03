# 🤝 Consensus Simulator

A multi-agent simulation framework that uses LLMs with **varied temperature settings** and **rich persona prompts** to simulate diverse people discussing questions and attempting to reach consensus.

## Overview

The Consensus Simulator creates a group of AI-powered "people," each with a unique:

- **Background** — age, occupation, education, life story
- **Personality** — thinking style, traits, communication preferences
- **Temperature** — LLM sampling temperature that controls response variability
- **Values & biases** — core beliefs that shape their opinions

These agents then engage in a **multi-round discussion** about a question, responding to each other's arguments, shifting positions, and ultimately trying to form consensus. An impartial facilitator analyzes the full discussion and produces a consensus report.

## Features

- **10 pre-built personas** — environmental scientist, small business owner, pastor, steelworker, tech founder, teacher, healthcare admin, journalist, economist, and organic farmer
- **Randomized agent generation** — factory system creates diverse groups from pools of names, occupations, traits, and values
- **Temperature spread** — agents are automatically assigned temperatures across a configurable range (e.g. 0.3–1.2), producing varied response styles
- **Multi-round discussions** — agents see what others said in previous rounds and can adjust their positions
- **Stance extraction** — automatic classification of each response as strongly for, somewhat for, neutral, somewhat against, or strongly against
- **Consensus detection** — checks if a threshold fraction of agents agree after each round
- **Facilitator analysis** — an impartial LLM generates a structured consensus report with areas of agreement/disagreement
- **Moderator summaries** — each round gets a concise summary
- **Rich terminal UI** — colored panels, tables, progress indicators via Rich
- **Output formats** — JSON results, Markdown transcripts, and summary reports
- **Configurable** — CLI flags, JSON config files, environment variables, or programmatic API

## Installation

```bash
# Clone or download the project
cd consensus-simulator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `openai` >= 1.30.0 — LLM API client
- `python-dotenv` >= 1.0.0 — environment variable loading
- `pydantic` >= 2.0.0 — configuration validation
- `rich` >= 13.0.0 — terminal formatting
- `typer` >= 0.9.0 — CLI framework
- `jinja2` >= 3.1.0 — template engine

### API Key Setup

Set your OpenAI API key via environment variable:

```bash
# Option 1: .env file
cp .env.example .env
# Edit .env and add your key

# Option 2: Export directly
export OPENAI_API_KEY="sk-your-key-here"

# Option 3: Pass via CLI flag
--api-key "sk-your-key-here"
```

## Quick Start

### CLI — Quick Mode

The fastest way to run a simulation:

```bash
python -m consensus_simulator quick "Should AI be regulated by governments?"
```

### CLI — Full Mode

With full control over parameters:

```bash
python -m consensus_simulator run \
  "Should universal basic income be implemented nationwide?" \
  --agents 6 \
  --rounds 4 \
  --model gpt-4 \
  --temp-min 0.3 \
  --temp-max 1.2 \
  --threshold 0.7 \
  --context "UBI proposals vary but generally involve providing all citizens with a regular unconditional sum of money." \
  --category economics \
  --output output \
  --seed 42
```

### CLI — Config File

```bash
# Generate a sample config
python -m consensus_simulator generate-config

# Edit the config, then run
python -m consensus_simulator --config sample_config.json
```

### CLI — Analyze Previous Results

```bash
python -m consensus_simulator analyze output/sim_should_ai_be_regulated_20240101_120000.json
```

### CLI — List Personas

```bash
python -m consensus_simulator list-personas
```

## Programmatic Usage

### Simple One-Call Simulation

```python
from consensus_simulator import quick_simulate

result = quick_simulate(
    question="Is a four-day work week beneficial for society?",
    num_agents=5,
    max_rounds=4,
    temperature_range=(0.3, 1.2),
    consensus_threshold=0.7,
    model="gpt-4",
    verbose=True,
    output_path="output/my_sim.json",
)

print(f"Consensus reached: {result.consensus_reached}")
print(f"Total rounds: {result.total_rounds}")
print(f"Analysis: {result.final_consensus}")
```

### Custom Agent Group

```python
from openai import OpenAI
from consensus_simulator.agents import AgentFactory, ThinkingStyle

client = OpenAI(api_key="your-key")

factory = AgentFactory(client=client, model="gpt-4", seed=42)

# Create a diverse group with specific temperature range
agents = factory.create_group(
    count=6,
    temperature_range=(0.3, 1.2),
    ensure_diversity=True,
)

# Or create agents with specific attributes
custom_agents = factory.create_custom_group([
    {"name": "Dr. Alice", "temperature": 0.3, "thinking_style": ThinkingStyle.ANALYTICAL},
    {"name": "Bob", "temperature": 1.1, "thinking_style": ThinkingStyle.CREATIVE},
    {"name": "Carol", "temperature": 0.7, "thinking_style": ThinkingStyle.EMOTIONAL},
    {"name": "Dave", "temperature": 0.9, "thinking_style": ThinkingStyle.SKEPTICAL},
])
```

### Full Simulation with Engine

```python
from openai import OpenAI
from consensus_simulator.agents import AgentFactory
from consensus_simulator.simulation import SimulationEngine

client = OpenAI(api_key="your-key")

# Create agents
factory = AgentFactory(client=client, model="gpt-4", seed=42)
agents = factory.create_group(count=5, temperature_range=(0.4, 1.0))

# Create engine
engine = SimulationEngine(
    client=client,
    model="gpt-4",
    max_rounds=5,
    consensus_threshold=0.7,
    parallel=True,
    verbose=True,
)

# Run simulation
result = engine.run(
    question="Should social media platforms be held liable for user content?",
    agents=agents,
    context="Section 230 of the Communications Decency Act shields platforms from liability.",
    category="policy",
)

# Save results
engine.save_result(result, "output/simulation.json")

# Access data
for rnd in result.rounds:
    print(f"\nRound {rnd.round_number}:")
    for resp in rnd.responses:
        print(f"  {resp.agent_name} (temp={resp.temperature}): {resp.stance}")
    print(f"  Distribution: {rnd.stance_distribution}")
```

### Using Pre-Built Personas

```python
from openai import OpenAI
from consensus_simulator.prompts import get_personas_by_indices, build_system_prompt
from consensus_simulator.agents import AgentFactory

# Select specific personas by index (see list-personas command)
personas = get_personas_by_indices([0, 2, 4, 6, 8])

for persona in personas:
    system_prompt = build_system_prompt(persona)
    print(f"--- {persona['name']} ---")
    print(system_prompt[:200] + "...\n")
```

## Architecture

### Project Structure

```
consensus-simulator/
├── consensus_simulator/          # Main package
│   ├── __init__.py               # Public API exports
│   ├── __main__.py               # python -m entry point
│   ├── main.py                   # CLI commands (Typer)
│   ├── agents.py                 # Agent class, profiles, factory
│   ├── prompts.py                # Persona templates, prompt builders
│   ├── simulation.py             # Simulation engine, consensus logic
│   ├── config.py                 # Pydantic configuration models
│   └── utils.py                  # Formatting, stance extraction, file I/O
├── output/                       # Generated results (gitignored)
├── sample_config.json            # Example configuration file
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
└── README.md                     # This file
```

### How It Works

```
┌─────────────────────────────────────────────────┐
│                  SIMULATION FLOW                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. SETUP                                        │
│     ┌──────────┐    ┌──────────────┐            │
│     │ Question │ +  │ Agent Group  │            │
│     └──────────┘    └──────┬───────┘            │
│                           │                      │
│         ┌─────────────────┼────────────┐        │
│         │                 │            │         │
│    ┌────▼───┐      ┌─────▼────┐  ┌───▼─────┐  │
│    │Agent A │      │ Agent B  │  │ Agent C  │  │
│    │temp=.3 │      │ temp=.7  │  │temp=1.1  │  │
│    │analyst │      │ pragmat. │  │ creative │  │
│    └────┬───┘      └─────┬────┘  └───┬─────┘  │
│         │                 │            │         │
│                                                  │
│  2. ROUND 1 — Initial Opinions                   │
│     Each agent responds to the question           │
│     Stances extracted from responses              │
│                                                  │
│  3. ROUNDS 2..N — Discussion                     │
│     Agents see what others said                   │
│     They may agree, disagree, or adjust           │
│     Moderator summarizes each round               │
│     Check for consensus after each round          │
│                                                  │
│  4. FINAL ANALYSIS                               │
│     Impartial facilitator reviews full transcript │
│     Produces structured consensus report          │
│                                                  │
│  5. OUTPUT                                       │
│     JSON results + Markdown transcript + summary  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Key Concepts

#### Temperature and Response Variation

Each agent is assigned a distinct temperature value spread across the configured range:

| Agent | Temperature | Effect |
|-------|------------|--------|
| Agent 1 | 0.3 | Highly deterministic, consistent, focused |
| Agent 2 | 0.5 | Moderate consistency, some variation |
| Agent 3 | 0.7 | Balanced creativity and focus |
| Agent 4 | 0.9 | More varied, creative, willing to explore |
| Agent 5 | 1.1 | Highly varied, unpredictable, creative |

Lower temperatures produce agents who stick firmly to their character's worldview. Higher temperatures produce agents who are more exploratory and may express more nuanced or unexpected opinions.

#### Persona System

Each agent's system prompt encodes:

1. **Identity** — name, age, occupation, education level
2. **Background** — a detailed life story paragraph
3. **Personality traits** — 2-4 descriptive traits
4. **Thinking style** — analytical, creative, pragmatic, emotional, skeptical, optimistic, conservative, or collaborative
5. **Communication style** — how they express themselves
6. **Values and biases** — core beliefs that shape opinions
7. **Expertise areas** — topics they know well

#### Stance Classification

Agent responses are classified into five categories:

- `strongly for` — enthusiastic support
- `somewhat for` — general agreement with reservations
- `neutral` — undecided, ambivalent, or mixed
- `somewhat against` — general opposition with openness
- `strongly against` — firm opposition

Classification uses `<stance>` tags in responses plus keyword heuristics as a fallback.

#### Consensus Detection

After each round, the engine checks whether the fraction of agents sharing a position (grouped as "for" or "against") meets the configured threshold:

```
consensus_threshold = 0.7  (default)

If >= 70% of agents are "for" (strongly + somewhat) → consensus reached
If >= 70% of agents are "against" (strongly + somewhat) → consensus reached
```

## CLI Reference

### `run` — Full Simulation

```bash
python -m consensus_simulator run [QUESTION] [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `QUESTION` | required | The discussion topic |
| `--config`, `-c` | — | Path to JSON config file |
| `--agents`, `-n` | 5 | Number of agents (2-20) |
| `--rounds`, `-r` | 4 | Max discussion rounds (1-20) |
| `--model`, `-m` | gpt-4 | LLM model identifier |
| `--api-key` | env var | OpenAI API key |
| `--api-base` | openai | API base URL |
| `--temp-min` | 0.3 | Minimum agent temperature |
| `--temp-max` | 1.2 | Maximum agent temperature |
| `--threshold`, `-t` | 0.7 | Consensus threshold (0.5-1.0) |
| `--output`, `-o` | output | Output directory |
| `--seed`, `-s` | random | Random seed |
| `--context` | — | Additional topic context |
| `--category` | general | Topic category label |
| `--parallel/--sequential` | parallel | Parallel agent queries |
| `--verbose/--quiet` | verbose | Console output detail |
| `--save-json/--no-save-json` | save | Save JSON results |
| `--save-markdown/--no-save-markdown` | save | Save Markdown files |

### `quick` — Fast Simulation

```bash
python -m consensus_simulator quick "Your question" --agents 5 --rounds 3
```

### `list-personas` — Show Available Personas

```bash
python -m consensus_simulator list-personas
```

### `generate-config` — Create Sample Config

```bash
python -m consensus_simulator generate-config --output my_config.json
```

### `analyze` — Review Saved Results

```bash
python -m consensus_simulator analyze output/sim_results.json
```

## Output Files

Each simulation generates up to three files in the output directory:

### JSON Results (`sim_*.json`)

Complete simulation data including all agent profiles, round-by-round responses, stance classifications, moderator summaries, and the final consensus analysis. Can be re-analyzed later with the `analyze` command.

### Markdown Transcript (`transcript_*.md`)

Human-readable transcript showing the full discussion organized by rounds, with stance labels and the consensus report at the end.

### Summary Report (`summary_*.md`)

Condensed report with:
- Agent stance table showing initial vs. final positions
- Stance distribution bar chart
- Consensus level and statement

## Configuration File

The JSON config file supports all CLI options:

```json
{
  "question": "Should AI be regulated?",
  "context": "Additional background...",
  "category": "technology",
  "num_agents": 6,
  "max_rounds": 4,
  "consensus_threshold": 0.7,
  "model": "gpt-4",
  "temperature_min": 0.3,
  "temperature_max": 1.2,
  "seed": 42,
  "parallel": true,
  "persona_indices": [0, 1, 2, 3, 4, 5],
  "output_settings": {
    "output_dir": "output",
    "save_json": true,
    "save_markdown": true,
    "verbose": true
  }
}
```

## Extending the Simulator

### Adding Custom Personas

Add new persona dictionaries to the `PERSONA_TEMPLATES` list in `consensus_simulator/prompts.py`:

```python
{
    "name": "Your Character Name",
    "age": 35,
    "occupation": "their job title",
    "background": "A paragraph about their life story...",
    "personality_traits": [
        "trait one",
        "trait two",
        "trait three",
    ],
    "communication_style": "How they speak and interact...",
    "biases": [
        "bias or perspective one",
        "bias or perspective two",
    ],
    "expertise_areas": [
        "area of knowledge one",
        "area of knowledge two",
    ],
}
```

### Using Different LLM Providers

Set `--api-base` to any OpenAI-compatible endpoint:

```bash
# Local model via Ollama
python -m consensus_simulator run "Your question" \
  --api-base http://localhost:11434/v1 \
  --model llama3

# Azure OpenAI
python -m consensus_simulator run "Your question" \
  --api-base https://your-resource.openai.azure.com/openai/deployments/your-deployment \
  --api-key your-azure-key
```

### Programmatic Customization

The `AgentFactory` supports creating agents with specific attributes:

```python
from consensus_simulator.agents import AgentFactory, ThinkingStyle, EducationLevel

factory = AgentFactory(client=client, model="gpt-4")

agents = factory.create_custom_group([
    {
        "name": "Dr. Logic",
        "temperature": 0.2,
        "thinking_style": ThinkingStyle.ANALYTICAL,
        "education": EducationLevel.DOCTORATE,
        "occupation": "Mathematician",
    },
    {
        "name": "Free Spirit",
        "temperature": 1.5,
        "thinking_style": ThinkingStyle.CREATIVE,
        "education": EducationLevel.SELF_TAUGHT,
        "occupation": "Artist",
    },
])
```

## Example Output

Running a simulation on "Should AI be regulated?" with 5 agents might produce:

```
╔══════════════════════════════════════════════════════════╗
║          🤝  CONSENSUS SIMULATOR  🤝                     ║
║   Simulate diverse perspectives reaching consensus       ║
╚══════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────┐
│  Question: Should AI be regulated by governments?        │
│  Model: gpt-4  |  Agents: 5  |  Rounds: 4               │
│  Temp range: [0.30, 1.20]  |  Threshold: 70%            │
└──────────────────────────────────────────────────────────┘

╭─ Dr. Sarah Chen (temp=0.30, analytical) ────────────────╮
│ Strong oversight is essential. Without regulation, AI    │
│ systems could cause real harm...                          │
│ Stance: strongly for                                     │
╰──────────────────────────────────────────────────────────╯

╭─ Marcus Williams (temp=0.60, pragmatic) ────────────────╮
│ Look, I'm all for safety, but we need to be careful      │
│ not to kill innovation...                                │
│ Stance: somewhat for                                     │
╰──────────────────────────────────────────────────────────╯

...

════════════════════════════════════════════════════════════
  📋 FINAL CONSENSUS ANALYSIS
════════════════════════════════════════════════════════════

Consensus Level: 4/5 — Near consensus
Consensus Reached: ✅ Yes

🤝 Areas of Agreement:
  • AI should face some form of oversight
  • Regulations should be risk-based and proportional
  • International coordination is important

⚡ Areas of Disagreement:
  • Speed and aggressiveness of implementation
  • Role of self-regulation vs. government mandates

📝 Consensus Statement:
The group broadly agrees that AI should be regulated through
a risk-based framework that balances innovation with safety...
```

## Requirements

- Python 3.10+
- OpenAI API key (or compatible API endpoint)
- Internet connection for API calls

## License

MIT License — feel free to use, modify, and distribute.