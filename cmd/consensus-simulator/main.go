package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/sashabaranov/go-openai"
	"github.com/spf13/cobra"

	"consensus-simulator/internal/simulator"
)

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

var (
	styleBold      = lipgloss.NewStyle().Bold(true)
	styleBoldBlue  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	styleBoldCyan  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("14"))
	styleBoldGreen = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	styleBoldRed   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("9"))
	styleBoldYellow = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11"))
	styleDim       = lipgloss.NewStyle().Faint(true)
	styleAgent     = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	styleFor       = lipgloss.NewStyle().Foreground(lipgloss.Color("10"))
	styleAgainst   = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	styleNeutral   = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
)

// ---------------------------------------------------------------------------
// Root command
// ---------------------------------------------------------------------------

var rootCmd = &cobra.Command{
	Use:   "consensus-simulator",
	Short: "Simulate a group of diverse people discussing topics to form consensus.",
	Long: `Consensus Simulator simulates diverse AI personas engaging in multi-round
discussions to reach consensus on questions. Each agent has a unique personality,
background, and temperature setting that produces varied, realistic responses.`,
}

// ---------------------------------------------------------------------------
// run command flags
// ---------------------------------------------------------------------------

var runFlags struct {
	config        string
	agents        int
	rounds        int
	model         string
	apiKey        string
	apiBase       string
	tempMin       float64
	tempMax       float64
	threshold     float64
	outputDir     string
	saveJSON      bool
	saveMarkdown  bool
	seed          int64
	verbose       bool
	parallel      bool
	context       string
	category      string
	personaIndices string
}

var runCmd = &cobra.Command{
	Use:   "run [question]",
	Short: "Run a consensus simulation with diverse AI-simulated participants.",
	Long: `Run a full consensus simulation where diverse AI agents discuss a question
over multiple rounds and attempt to reach consensus.

Each agent has a unique persona (background, personality, values) and
temperature setting, producing varied and realistic responses.`,
	Args: cobra.MaximumNArgs(1),
	RunE: runSimulation,
}

// ---------------------------------------------------------------------------
// quick command flags
// ---------------------------------------------------------------------------

var quickFlags struct {
	agents int
	rounds int
	model  string
	apiKey string
}

var quickCmd = &cobra.Command{
	Use:   "quick [question]",
	Short: "Quick simulation with minimal configuration.",
	Long:  "Run a fast consensus simulation with sensible defaults.",
	Args:  cobra.ExactArgs(1),
	RunE:  runQuick,
}

// ---------------------------------------------------------------------------
// list-personas command
// ---------------------------------------------------------------------------

var listPersonasCmd = &cobra.Command{
	Use:   "list-personas",
	Short: "List all available pre-defined persona templates.",
	RunE:  listPersonas,
}

// ---------------------------------------------------------------------------
// generate-config command
// ---------------------------------------------------------------------------

var generateConfigFlags struct {
	output string
}

var generateConfigCmd = &cobra.Command{
	Use:   "generate-config",
	Short: "Generate a sample configuration file.",
	RunE:  generateConfig,
}

// ---------------------------------------------------------------------------
// analyze command
// ---------------------------------------------------------------------------

var analyzeCmd = &cobra.Command{
	Use:   "analyze [path]",
	Short: "Analyze a previously saved simulation result.",
	Long:  "Load and display results from a previously saved simulation JSON file.",
	Args:  cobra.ExactArgs(1),
	RunE:  analyzeResults,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	// Register commands
	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(quickCmd)
	rootCmd.AddCommand(listPersonasCmd)
	rootCmd.AddCommand(generateConfigCmd)
	rootCmd.AddCommand(analyzeCmd)

	// run flags
	runCmd.Flags().StringVarP(&runFlags.config, "config", "c", "", "Path to a JSON configuration file")
	runCmd.Flags().IntVarP(&runFlags.agents, "agents", "n", 5, "Number of simulated agents (2-20)")
	runCmd.Flags().IntVarP(&runFlags.rounds, "rounds", "r", 4, "Maximum number of discussion rounds (1-20)")
	runCmd.Flags().StringVarP(&runFlags.model, "model", "m", "", "LLM model to use (e.g. gpt-4, gpt-3.5-turbo)")
	runCmd.Flags().StringVar(&runFlags.apiKey, "api-key", "", "OpenAI API key (or set OPENAI_API_KEY)")
	runCmd.Flags().StringVar(&runFlags.apiBase, "api-base", "", "OpenAI-compatible API base URL")
	runCmd.Flags().Float64Var(&runFlags.tempMin, "temp-min", 0.3, "Minimum temperature for agents (0.0-2.0)")
	runCmd.Flags().Float64Var(&runFlags.tempMax, "temp-max", 1.2, "Maximum temperature for agents (0.0-2.0)")
	runCmd.Flags().Float64VarP(&runFlags.threshold, "threshold", "t", 0.7, "Consensus threshold (0.5-1.0)")
	runCmd.Flags().StringVarP(&runFlags.outputDir, "output", "o", "output", "Directory to save results")
	runCmd.Flags().BoolVar(&runFlags.saveJSON, "save-json", true, "Save results as JSON")
	runCmd.Flags().BoolVar(&runFlags.saveMarkdown, "save-markdown", true, "Save transcript as Markdown")
	runCmd.Flags().Int64VarP(&runFlags.seed, "seed", "s", 0, "Random seed for reproducibility (0 = random)")
	runCmd.Flags().BoolVar(&runFlags.verbose, "verbose", true, "Enable verbose output")
	runCmd.Flags().BoolVar(&runFlags.parallel, "parallel", true, "Run agent queries in parallel")
	runCmd.Flags().StringVar(&runFlags.context, "context", "", "Additional context for the discussion topic")
	runCmd.Flags().StringVar(&runFlags.category, "category", "general", "Category label for the topic")
	runCmd.Flags().StringVar(&runFlags.personaIndices, "personas", "", "Comma-separated persona indices to use")

	// quick flags
	quickCmd.Flags().IntVarP(&quickFlags.agents, "agents", "n", 5, "Number of simulated agents (2-15)")
	quickCmd.Flags().IntVarP(&quickFlags.rounds, "rounds", "r", 3, "Maximum number of discussion rounds (1-10)")
	quickCmd.Flags().StringVarP(&quickFlags.model, "model", "m", "", "LLM model to use")
	quickCmd.Flags().StringVar(&quickFlags.apiKey, "api-key", "", "OpenAI API key (or set OPENAI_API_KEY)")

	// generate-config flags
	generateConfigCmd.Flags().StringVarP(&generateConfigFlags.output, "output", "o", "simulation_config.json", "Path to write the configuration file")

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func getAPIKey(provided string) (string, error) {
	key := provided
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	if key == "" {
		return "", fmt.Errorf("no API key provided; use --api-key or set OPENAI_API_KEY environment variable")
	}
	return key, nil
}

func getAPIBase(provided string) string {
	if provided != "" {
		return provided
	}
	if v := os.Getenv("OPENAI_API_BASE"); v != "" {
		return v
	}
	return "https://api.openai.com/v1"
}

func getModel(provided string) string {
	if provided != "" {
		return provided
	}
	if v := os.Getenv("OPENAI_MODEL"); v != "" {
		return v
	}
	return "gpt-4"
}

func printBanner() {
	box := lipgloss.NewStyle().
		Border(lipgloss.DoubleBorder()).
		BorderForeground(lipgloss.Color("12")).
		Padding(0, 2)

	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("14")).
		Render("🤝  CONSENSUS SIMULATOR  🤝")

	subtitle := lipgloss.NewStyle().
		Faint(true).
		Render("Simulate diverse perspectives reaching consensus")

	content := lipgloss.NewStyle().Align(lipgloss.Center).Render(
		title + "\n\n" + subtitle,
	)

	fmt.Println(box.Render(content))
}

func printAgentsTable(agents []*simulator.Agent) {
	// Header
	header := fmt.Sprintf("%-25s %-4s %-25s %-15s %-14s %-6s %s",
		"Name", "Age", "Occupation", "Education", "Thinking", "Temp", "Traits")
	fmt.Println(styleBold.Render(header))
	fmt.Println(strings.Repeat("─", 120))

	for _, a := range agents {
		traits := strings.Join(a.Profile.PersonalityTraits[:min(3, len(a.Profile.PersonalityTraits))], ", ")
		edu := strings.ReplaceAll(string(a.Profile.Education), "_", " ")
		line := fmt.Sprintf("%-25s %-4d %-25s %-15s %-14s %-6.2f %s",
			a.Profile.Name,
			a.Profile.Age,
			a.Profile.Occupation,
			edu,
			a.Profile.ThinkingStyle,
			a.Config.Temperature,
			traits,
		)
		fmt.Println(styleAgent.Render(line))
	}
	fmt.Println()
}

func stanceStyle(stance string) lipgloss.Style {
	if strings.Contains(stance, "for") {
		return styleFor
	}
	if strings.Contains(stance, "against") {
		return styleAgainst
	}
	return styleNeutral
}

func printRoundResults(roundNum, maxRounds int, responses []simulator.RoundResponse, stanceDist map[string]int) {
	fmt.Println()
	fmt.Println(styleBoldBlue.Render(fmt.Sprintf("━━━ Round %d / %d ━━━", roundNum, maxRounds)))

	for _, resp := range responses {
		stanceStr := resp.Stance
		if stanceStr == "" {
			stanceStr = "unclear"
		}

		box := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("8")).
			Padding(0, 2)

		content := resp.Text + "\n\n" +
			stanceStyle(stanceStr).Render("Stance: "+stanceStr) +
			styleDim.Render(fmt.Sprintf("  |  temp=%.2f  |  style=%s", resp.Temperature, resp.ThinkingStyle))

		title := styleAgent.Render(resp.AgentName)
		fmt.Println(box.Render(lipgloss.NewStyle().Render(title + "\n" + content)))
	}

	// Stance distribution
	fmt.Println()
	fmt.Print(styleBold.Render("Stance Distribution:  "))
	total := 0
	for _, count := range stanceDist {
		total += count
	}

	type stanceCount struct {
		stance string
		count  int
	}
	var sorted []stanceCount
	for s, c := range stanceDist {
		sorted = append(sorted, stanceCount{s, c})
	}
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].count > sorted[i].count {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	for _, sc := range sorted {
		pct := 0.0
		if total > 0 {
			pct = float64(sc.count) / float64(total) * 100
		}
		bar := strings.Repeat("█", int(pct/5))
		fmt.Println(stanceStyle(sc.stance).Render(fmt.Sprintf("\n  %s: %d (%.0f%%) %s", sc.stance, sc.count, pct, bar)))
	}
	fmt.Println()
}

func printConsensusResult(result *simulator.SimulationResult) {
	fmt.Println()
	fmt.Println(styleBoldGreen.Render("━━━ 📋 Final Consensus Analysis ━━━"))

	if result.FinalConsensus == nil {
		fmt.Println(styleBoldYellow.Render("No consensus analysis available."))
		return
	}

	fc := result.FinalConsensus

	// Consensus level
	level, _ := fc["consensus_level"].(string)
	if level == "" {
		level = "N/A"
	}
	levelStyle := styleDim
	// Try to parse numeric level
	if strings.Contains(level, "4") || strings.Contains(level, "5") {
		levelStyle = styleBoldGreen
	} else if strings.Contains(level, "3") {
		levelStyle = styleBoldYellow
	} else if strings.Contains(level, "1") || strings.Contains(level, "2") {
		levelStyle = styleBoldRed
	}

	fmt.Println()
	fmt.Println(styleBold.Render("Consensus Level:") + " " + levelStyle.Render(level+" / 5"))

	reachedText := "✅ Yes"
	if !result.ConsensusReached {
		reachedText = "❌ No"
	}
	fmt.Println(styleBold.Render("Consensus Reached:") + " " + reachedText)
	fmt.Println(styleBold.Render("Total Rounds:") + " " + fmt.Sprintf("%d", result.TotalRounds))

	// Areas of agreement
	if agreement, ok := fc["areas_of_agreement"]; ok {
		fmt.Println()
		fmt.Println(styleBoldGreen.Render("🤝 Areas of Agreement:"))
		if list, ok := agreement.([]interface{}); ok {
			for _, item := range list {
				fmt.Println(styleFor.Render("  • " + fmt.Sprintf("%v", item)))
			}
		} else {
			fmt.Println(styleFor.Render("  " + fmt.Sprintf("%v", agreement)))
		}
	}

	// Areas of disagreement
	if disagreement, ok := fc["areas_of_disagreement"]; ok {
		fmt.Println()
		fmt.Println(styleBoldRed.Render("⚡ Areas of Disagreement:"))
		if list, ok := disagreement.([]interface{}); ok {
			for _, item := range list {
				fmt.Println(styleAgainst.Render("  • " + fmt.Sprintf("%v", item)))
			}
		} else {
			fmt.Println(styleAgainst.Render("  " + fmt.Sprintf("%v", disagreement)))
		}
	}

	// Consensus statement
	if statement, ok := fc["consensus_statement"].(string); ok && statement != "" {
		box := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("14")).
			Padding(1, 2)
		fmt.Println()
		fmt.Println(box.Render(styleBold.Render("📝 Consensus Statement\n\n") + statement))
	}

	// Key perspectives
	if perspectives, ok := fc["key_perspectives"]; ok {
		fmt.Println()
		fmt.Println(styleBold.Render("🔑 Key Perspectives:"))
		if dict, ok := perspectives.(map[string]interface{}); ok {
			for name, perspective := range dict {
				fmt.Println(styleAgent.Render("  "+name+":") + " " + fmt.Sprintf("%v", perspective))
			}
		} else if list, ok := perspectives.([]interface{}); ok {
			for _, item := range list {
				fmt.Println("  • " + fmt.Sprintf("%v", item))
			}
		}
	}

	fmt.Println()
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

func runSimulation(cmd *cobra.Command, args []string) error {
	printBanner()

	question := ""
	if len(args) > 0 {
		question = args[0]
	}

	// Load from config file if provided
	if runFlags.config != "" {
		fmt.Println(styleDim.Render(fmt.Sprintf("Loading configuration from %s...", runFlags.config)))

		var cfg simulator.ConfigFileFormat
		if err := simulator.LoadJSON(runFlags.config, &cfg); err != nil {
			return fmt.Errorf("failed to load config: %w", err)
		}

		if question == "" {
			question = cfg.Question
		}
		if cfg.NumAgents > 0 {
			runFlags.agents = cfg.NumAgents
		}
		if cfg.MaxRounds > 0 {
			runFlags.rounds = cfg.MaxRounds
		}
		if cfg.ConsensusThreshold > 0 {
			runFlags.threshold = cfg.ConsensusThreshold
		}
		if cfg.Model != "" {
			runFlags.model = cfg.Model
		}
		if cfg.Context != "" {
			runFlags.context = cfg.Context
		}
		if cfg.Category != "" {
			runFlags.category = cfg.Category
		}
		if cfg.TemperatureMin > 0 {
			runFlags.tempMin = cfg.TemperatureMin
		}
		if cfg.TemperatureMax > 0 {
			runFlags.tempMax = cfg.TemperatureMax
		}
		if cfg.Seed != nil {
			runFlags.seed = *cfg.Seed
		}
		if runFlags.personaIndices == "" && len(cfg.PersonaIndices) > 0 {
			indices := make([]string, len(cfg.PersonaIndices))
			for i, idx := range cfg.PersonaIndices {
				indices[i] = fmt.Sprintf("%d", idx)
			}
			runFlags.personaIndices = strings.Join(indices, ",")
		}
		if cfg.OutputSettings.OutputDir != "" {
			runFlags.outputDir = cfg.OutputSettings.OutputDir
		}
		runFlags.saveJSON = cfg.OutputSettings.SaveJSON
		runFlags.saveMarkdown = cfg.OutputSettings.SaveMarkdown
		runFlags.verbose = cfg.OutputSettings.Verbose
	}

	// Validate question
	if question == "" {
		return fmt.Errorf("please provide a question or topic\n\nUsage: consensus-simulator run \"Your question here\"")
	}

	// Resolve settings
	apiKey, err := getAPIKey(runFlags.apiKey)
	if err != nil {
		return err
	}
	apiBase := getAPIBase(runFlags.apiBase)
	model := getModel(runFlags.model)

	// Print parameters
	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("14")).
		Padding(0, 2)

	seedStr := "random"
	if runFlags.seed != 0 {
		seedStr = fmt.Sprintf("%d", runFlags.seed)
	}

	params := fmt.Sprintf(
		"%s\n\n%s %s\n%s %s  |  %s %d  |  %s %d  |  %s %.0f%%\n%s [%.2f, %.2f]  |  %s %v  |  %s %s",
		styleBold.Render("Question: "+question),
		styleDim.Render("Model:"), model,
		styleDim.Render("Agents:"), runFlags.agents,
		styleDim.Render("Rounds:"), runFlags.rounds,
		styleDim.Render("Threshold:"), runFlags.threshold*100,
		styleDim.Render("Temp range:"), runFlags.tempMin, runFlags.tempMax,
		styleDim.Render("Parallel:"), runFlags.parallel,
		styleDim.Render("Seed:"), seedStr,
	)
	fmt.Println(box.Render("_simulation Parameters\n" + params))

	if runFlags.context != "" {
		fmt.Println(styleDim.Render("Context: " + runFlags.context))
		fmt.Println()
	}

	// Create client and agents
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = apiBase
	client := openai.NewClientWithConfig(config)

	var agents []*simulator.Agent
	var seed int64 = -1
	if runFlags.seed != 0 {
		seed = runFlags.seed
	}

	factory := simulator.NewAgentFactory(client, model, nil, seed)

	if runFlags.personaIndices != "" {
		// Use specific personas
		var indices []int
		for _, s := range strings.Split(runFlags.personaIndices, ",") {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			var idx int
			if _, err := fmt.Sscanf(s, "%d", &idx); err != nil {
				return fmt.Errorf("invalid persona index: %s", s)
			}
			indices = append(indices, idx)
		}

		personas, err := simulator.GetPersonasByIndices(indices)
		if err != nil {
			return fmt.Errorf("failed to get personas: %w", err)
		}
		agents = factory.CreateGroupFromPersonas(personas, runFlags.tempMin, runFlags.tempMax)
	} else {
		agents = factory.CreateGroup(runFlags.agents, runFlags.tempMin, runFlags.tempMax, true)
	}

	// Display agent roster
	fmt.Println(styleBoldYellow.Render("🧑‍🤝‍🧑 Simulated Participants"))
	printAgentsTable(agents)

	// Run simulation
	engine := simulator.NewSimulationEngine(client, model,
		simulator.WithMaxRounds(runFlags.rounds),
		simulator.WithConsensusThreshold(runFlags.threshold),
		simulator.WithParallel(runFlags.parallel),
		simulator.WithVerbose(false),
	)

	startTime := time.Now()
	fmt.Println(styleBoldCyan.Render("Running simulation..."))

	result := engine.Run(question, agents, runFlags.context, runFlags.category)

	elapsed := time.Since(startTime)

	// Display results round by round
	for _, rnd := range result.Rounds {
		printRoundResults(rnd.RoundNumber, runFlags.rounds, rnd.Responses, rnd.StanceDistribution)
		if rnd.ModeratorSummary != "" {
			box := lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("8")).
				Padding(0, 2)
			fmt.Println(box.Render(styleBold.Render("📝 Moderator Summary\n\n") + rnd.ModeratorSummary))
		}
	}

	// Display final consensus
	printConsensusResult(result)

	fmt.Println(styleDim.Render(fmt.Sprintf("Simulation completed in %.1f seconds.", elapsed.Seconds())))

	// Save outputs
	if err := os.MkdirAll(runFlags.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	safeQ := simulator.SafeFilename(question[:min(40, len(question))])
	ts := simulator.TimestampStr()

	if runFlags.saveJSON {
		jsonPath := filepath.Join(runFlags.outputDir, fmt.Sprintf("sim_%s_%s.json", safeQ, ts))
		if err := simulator.SaveResult(result, jsonPath); err != nil {
			return fmt.Errorf("failed to save JSON: %w", err)
		}
		fmt.Println(styleDim.Render("JSON results saved to: " + jsonPath))
	}

	if runFlags.saveMarkdown {
		// Build and save transcript
		var mdRounds []simulator.TranscriptRound
		for _, rnd := range result.Rounds {
			var tr simulator.TranscriptRound
			tr.RoundNumber = rnd.RoundNumber
			for _, r := range rnd.Responses {
				tr.Responses = append(tr.Responses, simulator.TranscriptResponse{
					Name:   r.AgentName,
					Text:   r.Text,
					Stance: r.Stance,
				})
			}
			mdRounds = append(mdRounds, tr)
		}

		transcriptMD := simulator.BuildTranscriptMarkdown(question, mdRounds, result.FinalConsensus)
		mdPath := filepath.Join(runFlags.outputDir, fmt.Sprintf("transcript_%s_%s.md", safeQ, ts))
		if err := simulator.SaveMarkdown(transcriptMD, mdPath); err != nil {
			return fmt.Errorf("failed to save transcript: %w", err)
		}
		fmt.Println(styleDim.Render("Markdown transcript saved to: " + mdPath))

		// Build and save summary
		var agentSummaries []simulator.AgentSummary
		if len(result.Rounds) > 0 {
			firstRound := result.Rounds[0]
			lastRound := result.Rounds[len(result.Rounds)-1]
			for _, firstResp := range firstRound.Responses {
				initialStance := firstResp.Stance
				if initialStance == "" {
					initialStance = "unclear"
				}
				finalStance := initialStance
				for _, lastResp := range lastRound.Responses {
					if lastResp.AgentName == firstResp.AgentName {
						finalStance = lastResp.Stance
						if finalStance == "" {
							finalStance = "unclear"
						}
						break
					}
				}
				agentSummaries = append(agentSummaries, simulator.AgentSummary{
					Name:           firstResp.AgentName,
					InitialStance:  initialStance,
					FinalStance:    finalStance,
					StanceChanged:  initialStance != finalStance,
				})
			}
		}

		summaryMD := simulator.BuildSummaryMarkdown(question, result.TotalRounds, agentSummaries, result.FinalConsensus)
		summaryPath := filepath.Join(runFlags.outputDir, fmt.Sprintf("summary_%s_%s.md", safeQ, ts))
		if err := simulator.SaveMarkdown(summaryMD, summaryPath); err != nil {
			return fmt.Errorf("failed to save summary: %w", err)
		}
		fmt.Println(styleDim.Render("Summary report saved to: " + summaryPath))
	}

	fmt.Println()
	fmt.Println(styleBoldGreen.Render("✅ Simulation complete!"))
	return nil
}

func runQuick(cmd *cobra.Command, args []string) error {
	printBanner()

	question := args[0]

	apiKey, err := getAPIKey(quickFlags.apiKey)
	if err != nil {
		return err
	}
	model := getModel(quickFlags.model)

	fmt.Println()
	fmt.Println(styleBold.Render("Question: " + question))
	fmt.Println(styleDim.Render(fmt.Sprintf("Agents: %d | Rounds: %d | Model: %s", quickFlags.agents, quickFlags.rounds, model)))
	fmt.Println()

	fmt.Println(styleBoldCyan.Render("Simulating..."))

	result, err := simulator.QuickSimulate(question, simulator.QuickSimulateOptions{
		NumAgents: quickFlags.agents,
		TempMin:   0.3,
		TempMax:   1.2,
		MaxRounds: quickFlags.rounds,
		Model:     model,
		APIKey:    apiKey,
		Verbose:   false,
	})
	if err != nil {
		return err
	}

	// Print brief results
	for _, rnd := range result.Rounds {
		fmt.Println(styleBoldBlue.Render(fmt.Sprintf("--- Round %d ---", rnd.RoundNumber)))
		for _, resp := range rnd.Responses {
			text := resp.Text
			if len(text) > 100 {
				text = text[:100] + "..."
			}
			stance := resp.Stance
			if stance == "" {
				stance = "unclear"
			}
			fmt.Printf("  %s: %s [%s]\n",
				styleAgent.Render(resp.AgentName),
				styleDim.Render(text),
				stanceStyle(stance).Render(stance),
			)
		}
	}

	printConsensusResult(result)
	return nil
}

func listPersonas(cmd *cobra.Command, args []string) error {
	printBanner()
	fmt.Println()
	fmt.Println(styleBold.Render("Available Persona Templates:"))
	fmt.Println()

	// Header
	fmt.Printf("%-4s %-25s %-4s %-35s %s\n",
		"#", "Name", "Age", "Occupation", "Traits")
	fmt.Println(strings.Repeat("─", 120))

	for i, p := range simulator.PersonaTemplates {
		traits := strings.Join(p.PersonalityTraits[:min(3, len(p.PersonalityTraits))], ", ")
		fmt.Printf("%-4d %-25s %-4d %-35s %s\n",
			i,
			styleAgent.Render(p.Name),
			p.Age,
			p.Occupation,
			traits,
		)
	}

	fmt.Println()
	fmt.Println(styleDim.Render("Use persona indices with --personas to select specific participants."))
	return nil
}

func generateConfig(cmd *cobra.Command, args []string) error {
	sampleConfig := simulator.ConfigFileFormat{
		Question:           "Should artificial intelligence be regulated by governments?",
		Context:            "AI systems are becoming increasingly powerful and autonomous. Some argue regulation stifles innovation while others believe it's essential for public safety.",
		Category:           "technology",
		NumAgents:          5,
		MaxRounds:          4,
		ConsensusThreshold: 0.7,
		Model:              "gpt-4",
		TemperatureMin:     0.3,
		TemperatureMax:     1.2,
		Seed:               nil,
		Parallel:           true,
		PersonaIndices:     []int{0, 1, 2, 3, 4},
		OutputSettings: simulator.OutputSettings{
			OutputDir:    "output",
			SaveJSON:     true,
			SaveMarkdown: true,
			Verbose:      true,
		},
	}

	data, err := json.MarshalIndent(sampleConfig, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(generateConfigFlags.output, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	fmt.Println(styleBoldGreen.Render("Sample configuration written to: " + generateConfigFlags.output))
	fmt.Println(styleDim.Render("Edit the file and run with --config <path>"))
	return nil
}

func analyzeResults(cmd *cobra.Command, args []string) error {
	printBanner()
	path := args[0]

	fmt.Println(styleDim.Render(fmt.Sprintf("Loading results from %s...", path)))
	fmt.Println()

	result, err := simulator.LoadResult(path)
	if err != nil {
		return fmt.Errorf("failed to load results: %w", err)
	}

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("14")).
		Padding(0, 2)

	reachedStr := "Yes"
	if !result.ConsensusReached {
		reachedStr = "No"
	}

	info := fmt.Sprintf(
		"%s\n\n%s %s\n%s Rounds: %d | Consensus: %s | Agents: %d",
		styleBold.Render("Question: "+result.Question),
		styleDim.Render("Rounds:"), result.TotalRounds,
		styleDim.Render(""), result.TotalRounds, reachedStr, len(result.AgentProfiles),
	)
	fmt.Println(box.Render("Simulation Results\n" + info))

	// Replay round results
	for _, rnd := range result.Rounds {
		printRoundResults(rnd.RoundNumber, result.TotalRounds, rnd.Responses, rnd.StanceDistribution)
	}

	// Consensus analysis
	printConsensusResult(result)

	// Stance evolution
	if len(result.Rounds) > 1 {
		fmt.Println(styleBold.Render("━━━ 📊 Stance Evolution ━━━"))

		// Collect all agent names
		agentNames := make(map[string]bool)
		for _, rnd := range result.Rounds {
			for _, resp := range rnd.Responses {
				agentNames[resp.AgentName] = true
			}
		}

		// Header
		fmt.Printf("%-25s", "Agent")
		for _, rnd := range result.Rounds {
			fmt.Printf(" %-15s", fmt.Sprintf("Round %d", rnd.RoundNumber))
		}
		fmt.Println()
		fmt.Println(strings.Repeat("─", 25+len(result.Rounds)*15))

		// Rows
		for name := range agentNames {
			fmt.Printf("%-25s", styleAgent.Render(name))
			for _, rnd := range result.Rounds {
				for _, resp := range rnd.Responses {
					if resp.AgentName == name {
						stance := resp.Stance
						if stance == "" {
							stance = "unclear"
						}
						fmt.Printf(" %-15s", stanceStyle(stance).Render(stance))
						break
					}
				}
			}
			fmt.Println()
		}
		fmt.Println()
	}

	return nil
}
