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
- **Styled terminal UI** — colored panels, tables, and indicators via lipgloss
- **Output formats** — JSON results, Markdown transcripts, and summary reports
- **Configurable** — CLI flags, JSON config files, environment variables, or programmatic API

## Installation

```bash
# Clone or download the project
cd consensus-simulator

# Build the CLI
go build -o consensus-simulator ./cmd/consensus-simulator

# Or install directly
go install ./cmd/consensus-simulator
```

### Dependencies

- `github.com/sashabaranov/go-openai` — LLM API client
- `github.com/spf13/cobra` — CLI framework
- `github.com/charmbracelet/lipgloss` — terminal styling

Dependencies are managed via Go modules. Run `go mod tidy` to fetch them.

### API Key Setup

Set your OpenAI API key via environment variable:

```bash
# Option 1: Export directly
export OPENAI_API_KEY="sk-your-key-here"

# Option 2: Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Option 3: Pass via CLI flag
--api-key "sk-your-key-here"
```

## Quick Start

### CLI — Quick Mode

```bash
consensus-simulator quick "Should AI be regulated by governments?"
```

### CLI — Full Mode

```bash
consensus-simulator run "Should AI be regulated by governments?" \
  --agents 6 \
  --rounds 4 \
  --model gpt-4 \
  --threshold 0.7
```

### CLI — Config File

```bash
# Generate a sample config
consensus-simulator generate-config --output my_config.json

# Edit the config, then run
consensus-simulator run --config my_config.json
```

### CLI — Analyze Previous Results

```bash
consensus-simulator analyze output/sim_should_ai_be_20240101_120000.json
```

### CLI — List Personas

```bash
consensus-simulator list-personas
```

## Programmatic Usage

### Simple One-Call Simulation

```go
package main

import (
    "fmt"
    "log"
    "consensus-simulator/internal/simulator"
)

func main() {
    result, err := simulator.QuickSimulate(
        "Should AI be regulated by governments?",
        simulator.QuickSimulateOptions{
            NumAgents:          5,
            MaxRounds:          4,
            ConsensusThreshold: 0.7,
            Model:              "gpt-4",
            Verbose:            true,
        },
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Consensus reached: %v\n", result.ConsensusReached)
    fmt.Printf("Total rounds: %d\n", result.TotalRounds)
}
```

### Custom Agent Group

```go
client := openai.NewClient("your-api-key")
factory := simulator.NewAgentFactory(client, "gpt-4", nil, 42)

agents := factory.CreateGroup(5, 0.3, 1.2, true)
for _, agent := range agents {
    fmt.Println(agent)
}
```

### Full Simulation with Engine

```go
client := openai.NewClient("your-api-key")
factory := simulator.NewAgentFactory(client, "gpt-4", nil, -1)
agents := factory.CreateGroup(6, 0.3, 1.2, true)

engine := simulator.NewSimulationEngine(client, "gpt-4",
    simulator.WithMaxRounds(4),
    simulator.WithConsensusThreshold(0.7),
    simulator.WithParallel(true),
)

result := engine.Run(
    "Should AI be regulated?",
    agents,
    "AI systems are becoming increasingly powerful.",
    "technology",
)

// Save results
err := simulator.SaveResult(result, "output/my_sim.json")
```

### Using Pre-Built Personas

```go
// Get personas by index
personas, err := simulator.GetPersonasByIndices([]int{0, 1, 3, 5, 7})
if err != nil {
    log.Fatal(err)
}

// Create agents from personas
agents := factory.CreateGroupFromPersonas(personas, 0.3, 1.2)

// Or get random personas
randomPersonas, err := simulator.GetRandomPersonas(4, 12345)
```

## Architecture

### Project Structure

```
consensus-simulator/
├── cmd/
│   └── consensus-simulator/
│       └── main.go            # CLI entry point (cobra commands)
├── internal/
│   └── simulator/
│       ├── types.go           # Core types, enums, config structures
│       ├── agents.go          # Agent struct and AgentFactory
│       ├── prompts.go         # Prompt templates and persona definitions
│       ├── simulation.go      # SimulationEngine and QuickSimulate
│       └── utils.go           # Stance extraction, formatting, file I/O
├── go.mod
├── go.sum
├── sample_config.json
└── README.md
```

### How It Works

1. **Agent Creation**: The `AgentFactory` creates diverse agents, either from pre-built personas or by randomly combining names, occupations, traits, and values. Each agent gets a unique temperature from the configured range.

2. **System Prompts**: Each agent receives a detailed system prompt encoding their persona — background, personality, communication style, biases, and expertise areas.

3. **Multi-Round Discussion**: The `SimulationEngine` orchestrates discussion rounds. In each round:
   - Agents receive the discussion topic and previous responses
   - Responses are gathered in parallel (or sequentially)
   - Stances are extracted from responses
   - A moderator summary is generated

4. **Consensus Detection**: After each round, the system checks if a threshold fraction of agents share the same position (for/against/neutral).

5. **Final Analysis**: An impartial LLM facilitator reviews the full transcript and produces a structured consensus report.

### Key Concepts

#### Temperature and Response Variation

Temperature controls how deterministic vs. creative an agent's responses are:
- **Low temperature (0.3)**: Analytical, consistent, careful — good for experts
- **Medium temperature (0.7)**: Balanced, natural conversation style
- **High temperature (1.2)**: Creative, varied, sometimes surprising — good for divergent thinkers

By spreading temperatures across a group, you get natural variation in how people approach a topic.

#### Persona System

10 pre-built personas cover diverse demographics and perspectives:
- Dr. Sarah Chen (environmental scientist)
- Marcus Williams (small business owner)
- Rev. Aisha Johnson (pastor and social worker)
- James "Jim" O'Brien (retired steelworker)
- Priya Patel (tech startup co-founder)
- Roberto Gutierrez (history teacher)
- Dr. Karen Whitfield (healthcare administrator)
- Tyler Nash (journalist)
- Mei-Lin Zhao (economist)
- Cody Blackwood (organic farmer)

Each persona includes detailed background, personality traits, communication style, biases, and expertise areas.

#### Stance Classification

Responses are classified into six categories:
- `strongly_for` — enthusiastic support
- `somewhat_for` — general agreement
- `neutral` — undecided or balanced view
- `somewhat_against` — general disagreement
- `strongly_against` — firm opposition
- `unclear` — cannot determine stance

Classification first checks for `<stance>` tags in the response, then falls back to keyword matching.

#### Consensus Detection

Consensus is reached when a threshold fraction (default 70%) of agents share the same broad position (for, against, or neutral). The "for" category combines both "strongly for" and "somewhat for" stances.

## CLI Reference

### `run` — Full Simulation

```bash
consensus-simulator run [question] [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config`, `-c` | | Path to JSON configuration file |
| `--agents`, `-n` | 5 | Number of simulated agents (2-20) |
| `--rounds`, `-r` | 4 | Maximum discussion rounds (1-20) |
| `--model`, `-m` | gpt-4 | LLM model to use |
| `--api-key` | | OpenAI API key |
| `--api-base` | | API base URL |
| `--temp-min` | 0.3 | Minimum temperature |
| `--temp-max` | 1.2 | Maximum temperature |
| `--threshold`, `-t` | 0.7 | Consensus threshold (0.5-1.0) |
| `--output`, `-o` | output | Output directory |
| `--save-json` | true | Save JSON results |
| `--save-markdown` | true | Save Markdown transcript |
| `--seed`, `-s` | 0 | Random seed (0 = random) |
| `--verbose` | true | Verbose output |
| `--parallel` | true | Parallel agent queries |
| `--context` | | Additional context |
| `--category` | general | Topic category |
| `--personas` | | Comma-separated persona indices |

### `quick` — Fast Simulation

```bash
consensus-simulator quick [question] [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agents`, `-n` | 5 | Number of agents (2-15) |
| `--rounds`, `-r` | 3 | Number of rounds (1-10) |
| `--model`, `-m` | gpt-4 | LLM model |
| `--api-key` | | OpenAI API key |

### `list-personas` — Show Available Personas

```bash
consensus-simulator list-personas
```

### `generate-config` — Create Sample Config

```bash
consensus-simulator generate-config [flags]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | simulation_config.json | Output path |

### `analyze` — Review Saved Results

```bash
consensus-simulator analyze [path]
```

## Output Files

### JSON Results (`sim_*.json`)

Complete simulation data including all rounds, responses, stances, and consensus analysis.

### Markdown Transcript (`transcript_*.md`)

Full discussion transcript with each agent's responses organized by round.

### Summary Report (`summary_*.md`)

Concise summary showing stance changes, final distribution, and consensus result.

## Configuration File

Example `simulation_config.json`:

```json
{
  "question": "Should artificial intelligence be regulated by governments?",
  "context": "AI systems are becoming increasingly powerful and autonomous.",
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

Add to `PersonaTemplates` in `internal/simulator/prompts.go`:

```go
{
    Name: "Dr. New Persona",
    Age: 35,
    Occupation: "data scientist",
    Background: "Your background story here...",
    PersonalityTraits: []string{"Analytical", "Curious"},
    CommunicationStyle: "How they speak...",
    Biases: []string{"Pro-data", "Skeptical of anecdotes"},
    ExpertiseAreas: []string{"Machine learning", "Statistics"},
}
```

### Using Different LLM Providers

Use `--api-base` to point to any OpenAI-compatible API:

```bash
consensus-simulator run "Your question" --api-base https://api.your-provider.com/v1
```

### Programmatic Customization

```go
opts := simulator.CreateAgentOptions{
    Name:          strPtr("Custom Agent"),
    Age:           intPtr(30),
    Occupation:    strPtr("Philosopher"),
    Temperature:   float64Ptr(0.5),
    ThinkingStyle: thinkingStylePtr(simulator.ThinkingAnalytical),
}
agent := factory.CreateAgent(opts)
```

## Example Output

```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🤝  CONSENSUS SIMULATOR  🤝                     ║
║                                                          ║
║   Simulate diverse perspectives reaching consensus       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

━━━ Round 1 / 4 ━━━

╭─────────────────────────────────────────╮
│ Dr. Sarah Chen                          │
│                                         │
│ As an environmental scientist...        │
│                                         │
│ Stance: strongly for  |  temp=0.30      │
╰─────────────────────────────────────────╯

...

━━━ 📋 Final Consensus Analysis ━━━

Consensus Level: 3 / 5
Consensus Reached: ❌ No
Total Rounds: 4

🤝 Areas of Agreement:
  • AI poses real risks that need addressing
  • Some form of oversight is necessary

⚡ Areas of Disagreement:
  • Scope and nature of regulation
  • Impact on innovation

📝 Consensus Statement
The group broadly agrees that some AI oversight is needed but
disagrees significantly on the appropriate scope and mechanism.
```

## Requirements

- Go 1.21 or later
- OpenAI API key (or compatible LLM provider)

## License

MIT