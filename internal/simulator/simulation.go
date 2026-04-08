package simulator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sashabaranov/go-openai"
)

// SimulationEngine drives a complete consensus simulation.
//
// Typical usage:
//
//	engine := NewSimulationEngine(client, "gpt-4", WithMaxRounds(5))
//	result := engine.Run("Should AI be regulated?", agents)
//	SaveResult(result, "output/my_sim.json")
type SimulationEngine struct {
	client             *openai.Client
	model              string
	maxRounds          int
	consensusThreshold float64
	parallel           bool
	verbose            bool
}

// SimulationOption configures a SimulationEngine.
type SimulationOption func(*SimulationEngine)

// WithMaxRounds sets the maximum number of discussion rounds.
func WithMaxRounds(n int) SimulationOption {
	return func(e *SimulationEngine) { e.maxRounds = n }
}

// WithConsensusThreshold sets the fraction of agents that must agree.
func WithConsensusThreshold(t float64) SimulationOption {
	return func(e *SimulationEngine) { e.consensusThreshold = t }
}

// WithParallel enables or disables parallel LLM requests.
func WithParallel(p bool) SimulationOption {
	return func(e *SimulationEngine) { e.parallel = p }
}

// WithVerbose enables or disables verbose logging.
func WithVerbose(v bool) SimulationOption {
	return func(e *SimulationEngine) { e.verbose = v }
}

// NewSimulationEngine creates a SimulationEngine with the given OpenAI client
// and model, applying any optional configuration.
func NewSimulationEngine(client *openai.Client, model string, opts ...SimulationOption) *SimulationEngine {
	e := &SimulationEngine{
		client:             client,
		model:              model,
		maxRounds:          5,
		consensusThreshold: 0.7,
		parallel:           true,
		verbose:            false,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// log prints a timestamped message if verbose mode is enabled.
func (e *SimulationEngine) log(msg string) {
	if e.verbose {
		ts := time.Now().UTC().Format("15:04:05")
		fmt.Printf("[%s] %s\n", ts, msg)
	}
}

// ---------------------------------------------------------------------------
// Single-agent response via prompts.go templates
// ---------------------------------------------------------------------------

// getAgentResponseWithTemplate generates an agent's response using the
// structured prompt templates.
func (e *SimulationEngine) getAgentResponseWithTemplate(
	ctx context.Context,
	agent *Agent,
	question string,
	roundNumber int,
	discussionHistory string,
	contextStr string,
) string {
	// Build a persona dict compatible with prompts.py
	persona := AgentPersona{
		Name:               agent.Profile.Name,
		Age:                agent.Profile.Age,
		Occupation:         agent.Profile.Occupation,
		Background:         agent.Profile.Background,
		PersonalityTraits:  agent.Profile.PersonalityTraits,
		CommunicationStyle: agent.Profile.CommunicationStyle,
		Biases:             safeSlice(agent.Profile.Values, 3),
		ExpertiseAreas:     safeSlice(agent.Profile.Values, 2),
	}

	systemPrompt := BuildSystemPrompt(persona)

	var userPrompt string
	if roundNumber == 1 {
		userPrompt = BuildInitialPrompt(question, persona)
	} else {
		userPrompt = BuildRoundPrompt(question, persona, roundNumber, discussionHistory)
	}

	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
		{Role: openai.ChatMessageRoleUser, Content: userPrompt},
	}

	resp, err := e.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:            e.model,
		Messages:         messages,
		Temperature:      float32(agent.Config.Temperature),
		MaxTokens:        agent.Config.MaxTokens,
		TopP:             float32(agent.Config.TopP),
		FrequencyPenalty: float32(agent.Config.FrequencyPenalty),
		PresencePenalty:  float32(agent.Config.PresencePenalty),
	})

	if err != nil {
		return fmt.Sprintf("[Error generating response: %v]", err)
	}
	if len(resp.Choices) == 0 {
		return "[Error: no response from LLM]"
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

// safeSlice returns the first n elements of s, or all elements if n > len(s).
func safeSlice(s []string, n int) []string {
	if n >= len(s) {
		return s
	}
	return s[:n]
}

// ---------------------------------------------------------------------------
// Moderator summary
// ---------------------------------------------------------------------------

// generateModeratorSummary asks the LLM to summarise a round as an impartial moderator.
func (e *SimulationEngine) generateModeratorSummary(
	ctx context.Context,
	question string,
	roundNumber int,
	responses []RoundResponse,
) string {
	var roundParts []string
	for _, r := range responses {
		roundParts = append(roundParts, fmt.Sprintf("**%s** (temp=%.2f):\n%s", r.AgentName, r.Temperature, r.Text))
	}
	roundText := strings.Join(roundParts, "\n\n")

	prompt := BuildModeratorSummaryPrompt(question, roundNumber, roundText)

	resp, err := e.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: e.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "You are an impartial discussion moderator. Summarise concisely."},
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		Temperature: 0.3,
		MaxTokens:   256,
	})

	if err != nil {
		return fmt.Sprintf("[Could not generate moderator summary: %v]", err)
	}
	if len(resp.Choices) == 0 {
		return "[Could not generate moderator summary: no response]"
	}
	return strings.TrimSpace(resp.Choices[0].Message.Content)
}

// ---------------------------------------------------------------------------
// Consensus analysis
// ---------------------------------------------------------------------------

var jsonFenceRegex = regexp.MustCompile("(?s)^```(?:json)?\\s*(.*?)\\s*```$")

// generateConsensusAnalysis uses the consensus prompt to produce a final analysis.
func (e *SimulationEngine) generateConsensusAnalysis(
	ctx context.Context,
	question string,
	transcript string,
) map[string]interface{} {
	prompt := BuildConsensusPrompt(question, transcript)

	resp, err := e.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: e.model,
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "You are an impartial facilitator analysing a group discussion. Respond ONLY with valid JSON."},
			{Role: openai.ChatMessageRoleUser, Content: prompt},
		},
		Temperature: 0.2,
		MaxTokens:   1024,
	})

	if err != nil {
		return map[string]interface{}{
			"consensus_level":     "error",
			"error":               err.Error(),
			"consensus_statement": "Analysis generation failed.",
		}
	}

	if len(resp.Choices) == 0 {
		return map[string]interface{}{
			"consensus_level":     "error",
			"error":               "no response",
			"consensus_statement": "Analysis generation failed.",
		}
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)

	// Strip markdown fences if present
	if strings.HasPrefix(raw, "```") {
		if match := jsonFenceRegex.FindStringSubmatch(raw); len(match) > 1 {
			raw = match[1]
		}
	}

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return map[string]interface{}{
			"consensus_level":       "unknown",
			"areas_of_agreement":    []interface{}{},
			"areas_of_disagreement": []interface{}{},
			"consensus_statement":   raw,
			"key_perspectives":      map[string]interface{}{},
			"raw_response":          raw,
		}
	}

	return result
}

// ---------------------------------------------------------------------------
// Discussion history helper
// ---------------------------------------------------------------------------

// FormatHistory builds a readable discussion history from previous rounds.
// If upToRound is -1, all rounds are included.
func FormatHistory(rounds []RoundResult, upToRound int) string {
	var target []RoundResult
	if upToRound < 0 {
		target = rounds
	} else if upToRound <= len(rounds) {
		target = rounds[:upToRound]
	}

	var parts []string
	for _, rnd := range target {
		parts = append(parts, fmt.Sprintf("--- Round %d ---", rnd.RoundNumber))
		for _, resp := range rnd.Responses {
			parts = append(parts, fmt.Sprintf("%s: %s", resp.AgentName, resp.Text))
		}
		parts = append(parts, "")
	}
	return strings.Join(parts, "\n")
}

// ---------------------------------------------------------------------------
// Main simulation loop
// ---------------------------------------------------------------------------

// Run executes a full multi-round consensus simulation.
//
// Parameters:
//   - question: The question or topic to discuss.
//   - agents: The simulated participants.
//   - context: Additional background information (optional).
//   - category: Topic category label.
//
// Returns a SimulationResult containing all rounds, stances, and final analysis.
func (e *SimulationEngine) Run(
	question string,
	agents []*Agent,
	contextStr string,
	category string,
) *SimulationResult {
	ctx := context.Background()
	startedAt := time.Now().UTC().Format(time.RFC3339)

	e.log(fmt.Sprintf("Starting simulation: %q...", truncate(question, 80)))
	e.log(fmt.Sprintf("Agents: %d | Max rounds: %d | Threshold: %.2f", len(agents), e.maxRounds, e.consensusThreshold))

	result := NewSimulationResult(question)
	result.Context = contextStr
	result.Category = category
	result.StartedAt = startedAt

	// Capture agent profiles
	for _, a := range agents {
		result.AgentProfiles = append(result.AgentProfiles, map[string]interface{}{
			"name":               a.Profile.Name,
			"age":                a.Profile.Age,
			"occupation":         a.Profile.Occupation,
			"education":          string(a.Profile.Education),
			"thinking_style":     string(a.Profile.ThinkingStyle),
			"temperature":        a.Config.Temperature,
			"personality_traits": a.Profile.PersonalityTraits,
			"values":             a.Profile.Values,
		})
	}

	consensusReached := false

	for roundNum := 1; roundNum <= e.maxRounds; roundNum++ {
		e.log(fmt.Sprintf("--- Round %d/%d ---", roundNum, e.maxRounds))
		roundResult := NewRoundResult(roundNum)

		// Build discussion history
		history := ""
		if len(result.Rounds) > 0 {
			history = FormatHistory(result.Rounds, -1)
		}

		// Gather responses — optionally in parallel
		var roundResponses []RoundResponse
		if e.parallel && len(agents) > 1 {
			roundResponses = e.gatherParallel(ctx, agents, question, roundNum, history, contextStr)
		} else {
			roundResponses = e.gatherSequential(ctx, agents, question, roundNum, history, contextStr)
		}

		roundResult.Responses = roundResponses

		// Extract stances
		for i := range roundResponses {
			roundResponses[i].Stance = ExtractStance(roundResponses[i].Text)
			e.log(fmt.Sprintf("  %s: stance=%s", roundResponses[i].AgentName, roundResponses[i].Stance))
		}

		// Compute distribution
		roundResult.StanceDistribution = ComputeStanceDistribution(roundResponses)
		e.log(fmt.Sprintf("  Distribution: %v", roundResult.StanceDistribution))

		// Generate moderator summary
		roundResult.ModeratorSummary = e.generateModeratorSummary(ctx, question, roundNum, roundResponses)
		e.log(fmt.Sprintf("  Summary: %s...", truncate(roundResult.ModeratorSummary, 120)))

		// Check consensus
		reached, dominant := CheckConsensus(roundResponses, e.consensusThreshold)
		roundResult.ConsensusReached = reached

		result.Rounds = append(result.Rounds, roundResult)

		if reached {
			consensusReached = true
			e.log(fmt.Sprintf("  ✓ Consensus reached! Dominant position: %s", dominant))
			if roundNum < e.maxRounds {
				e.log("  Running confirmation round...")
			} else {
				break
			}
		} else if roundNum == e.maxRounds {
			e.log("  Max rounds reached without consensus.")
		} else {
			e.log("  No consensus yet, continuing...")
		}
	}

	// Build transcript
	result.Transcript = BuildTranscript(result.Rounds)
	result.TotalRounds = len(result.Rounds)
	result.ConsensusReached = consensusReached

	// Generate final consensus analysis
	e.log("Generating final consensus analysis...")
	result.FinalConsensus = e.generateConsensusAnalysis(ctx, question, result.Transcript)

	result.FinishedAt = time.Now().UTC().Format(time.RFC3339)
	e.log("Simulation complete.")
	return &result
}

// ---------------------------------------------------------------------------
// Parallel / sequential gathering
// ---------------------------------------------------------------------------

// gatherParallel gathers agent responses using goroutines for parallelism.
func (e *SimulationEngine) gatherParallel(
	ctx context.Context,
	agents []*Agent,
	question string,
	roundNumber int,
	history string,
	contextStr string,
) []RoundResponse {
	type agentResult struct {
		index int
		resp  RoundResponse
		err   error
	}

	resultCh := make(chan agentResult, len(agents))
	var wg sync.WaitGroup

	for i, agent := range agents {
		wg.Add(1)
		go func(idx int, a *Agent) {
			defer wg.Done()
			text := e.getAgentResponseWithTemplate(ctx, a, question, roundNumber, history, contextStr)
			resultCh <- agentResult{
				index: idx,
				resp: RoundResponse{
					AgentName:     a.Profile.Name,
					RoundNumber:   roundNumber,
					Temperature:   a.Config.Temperature,
					ThinkingStyle: string(a.Profile.ThinkingStyle),
					Text:          text,
				},
			}
		}(i, agent)
	}

	wg.Wait()
	close(resultCh)

	var results []agentResult
	for r := range resultCh {
		results = append(results, r)
	}

	// Sort by original index for deterministic ordering
	sort.Slice(results, func(i, j int) bool {
		return results[i].index < results[j].index
	})

	responses := make([]RoundResponse, len(results))
	for i, r := range results {
		if r.err != nil {
			responses[i] = RoundResponse{
				AgentName:     agents[r.index].Profile.Name,
				RoundNumber:   roundNumber,
				Temperature:   agents[r.index].Config.Temperature,
				ThinkingStyle: string(agents[r.index].Profile.ThinkingStyle),
				Text:          fmt.Sprintf("[Error: %v]", r.err),
			}
		} else {
			responses[i] = r.resp
		}
	}

	return responses
}

// gatherSequential gathers agent responses one at a time.
func (e *SimulationEngine) gatherSequential(
	ctx context.Context,
	agents []*Agent,
	question string,
	roundNumber int,
	history string,
	contextStr string,
) []RoundResponse {
	responses := make([]RoundResponse, 0, len(agents))

	for _, agent := range agents {
		e.log(fmt.Sprintf("  Querying %s...", agent.Profile.Name))

		text := e.getAgentResponseWithTemplate(ctx, agent, question, roundNumber, history, contextStr)

		responses = append(responses, RoundResponse{
			AgentName:     agent.Profile.Name,
			RoundNumber:   roundNumber,
			Temperature:   agent.Config.Temperature,
			ThinkingStyle: string(agent.Profile.ThinkingStyle),
			Text:          text,
		})
	}

	return responses
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

// SaveResult saves a simulation result to a JSON file.
func SaveResult(result *SimulationResult, path string) error {
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}

	data := resultToDict(result)
	return SaveJSON(data, path)
}

// LoadResult loads a simulation result from a JSON file.
func LoadResult(path string) (*SimulationResult, error) {
	var data map[string]interface{}
	if err := LoadJSON(path, &data); err != nil {
		return nil, err
	}

	result := &SimulationResult{
		Question:         getString(data, "question"),
		Context:          getString(data, "context"),
		Category:         getString(data, "category"),
		TotalRounds:      getInt(data, "total_rounds"),
		ConsensusReached: getBool(data, "consensus_reached"),
		Transcript:       getString(data, "transcript"),
		StartedAt:        getString(data, "started_at"),
		FinishedAt:       getString(data, "finished_at"),
		Rounds:           make([]RoundResult, 0),
		AgentProfiles:    make([]map[string]interface{}, 0),
	}

	if fc, ok := data["final_consensus"].(map[string]interface{}); ok {
		result.FinalConsensus = fc
	}

	if ap, ok := data["agent_profiles"].([]interface{}); ok {
		for _, p := range ap {
			if m, ok := p.(map[string]interface{}); ok {
				result.AgentProfiles = append(result.AgentProfiles, m)
			}
		}
	}

	if rounds, ok := data["rounds"].([]interface{}); ok {
		for _, r := range rounds {
			if rndData, ok := r.(map[string]interface{}); ok {
				rnd := RoundResult{
					RoundNumber:        getInt(rndData, "round_number"),
					ModeratorSummary:   getString(rndData, "moderator_summary"),
					ConsensusReached:   getBool(rndData, "consensus_reached"),
					StanceDistribution: make(map[string]int),
				}

				if sd, ok := rndData["stance_distribution"].(map[string]interface{}); ok {
					for k, v := range sd {
						rnd.StanceDistribution[k] = toInt(v)
					}
				}

				if resps, ok := rndData["responses"].([]interface{}); ok {
					for _, resp := range resps {
						if rd, ok := resp.(map[string]interface{}); ok {
							rnd.Responses = append(rnd.Responses, RoundResponse{
								AgentName:     getString(rd, "agent_name"),
								RoundNumber:   getInt(rd, "round_number"),
								Temperature:   getFloat(rd, "temperature"),
								ThinkingStyle: getString(rd, "thinking_style"),
								Text:          getString(rd, "text"),
								Stance:        getString(rd, "stance"),
								Timestamp:     getString(rd, "timestamp"),
							})
						}
					}
				}

				result.Rounds = append(result.Rounds, rnd)
			}
		}
	}

	return result, nil
}

// resultToDict converts a SimulationResult to a plain map for JSON serialization.
func resultToDict(result *SimulationResult) map[string]interface{} {
	rounds := make([]interface{}, len(result.Rounds))
	for i, r := range result.Rounds {
		responses := make([]interface{}, len(r.Responses))
		for j, resp := range r.Responses {
			responses[j] = map[string]interface{}{
				"agent_name":     resp.AgentName,
				"round_number":   resp.RoundNumber,
				"temperature":    resp.Temperature,
				"thinking_style": resp.ThinkingStyle,
				"text":           resp.Text,
				"stance":         resp.Stance,
				"timestamp":      resp.Timestamp,
			}
		}

		stanceDist := make(map[string]interface{})
		for k, v := range r.StanceDistribution {
			stanceDist[k] = v
		}

		rounds[i] = map[string]interface{}{
			"round_number":        r.RoundNumber,
			"moderator_summary":   r.ModeratorSummary,
			"consensus_reached":   r.ConsensusReached,
			"stance_distribution": stanceDist,
			"responses":           responses,
		}
	}

	return map[string]interface{}{
		"question":          result.Question,
		"context":           result.Context,
		"category":          result.Category,
		"total_rounds":      result.TotalRounds,
		"consensus_reached": result.ConsensusReached,
		"final_consensus":   result.FinalConsensus,
		"transcript":        result.Transcript,
		"agent_profiles":    result.AgentProfiles,
		"started_at":        result.StartedAt,
		"finished_at":       result.FinishedAt,
		"rounds":            rounds,
	}
}

// ---------------------------------------------------------------------------
// Convenience: run simulation from simple parameters
// ---------------------------------------------------------------------------

// QuickSimulateOptions contains options for the QuickSimulate function.
type QuickSimulateOptions struct {
	NumAgents          int
	TempMin            float64
	TempMax            float64
	MaxRounds          int
	ConsensusThreshold float64
	Model              string
	APIKey             string
	APIBase            string
	Seed               int64
	Verbose            bool
	OutputPath         string
}

// QuickSimulate is a one-call convenience function for running a simulation.
// It creates a random diverse group of agents and runs the full discussion.
func QuickSimulate(question string, opts QuickSimulateOptions) (*SimulationResult, error) {
	key := opts.APIKey
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	if key == "" {
		return nil, fmt.Errorf("no API key provided; set api_key parameter or OPENAI_API_KEY env var")
	}

	base := opts.APIBase
	if base == "" {
		base = os.Getenv("OPENAI_API_BASE")
		if base == "" {
			base = "https://api.openai.com/v1"
		}
	}

	model := opts.Model
	if model == "" {
		model = os.Getenv("OPENAI_MODEL")
		if model == "" {
			model = "gpt-4"
		}
	}

	config := openai.DefaultConfig(key)
	config.BaseURL = base
	client := openai.NewClientWithConfig(config)

	var seed int64 = -1
	if opts.Seed != 0 {
		seed = opts.Seed
	}

	factory := NewAgentFactory(client, model, nil, seed)
	agents := factory.CreateGroup(opts.NumAgents, opts.TempMin, opts.TempMax, true)

	if opts.Verbose {
		fmt.Printf("\n%s\n", strings.Repeat("=", 60))
		fmt.Println("  CONSENSUS SIMULATION")
		fmt.Printf("%s\n", strings.Repeat("=", 60))
		fmt.Printf("  Question: %s\n", question)
		fmt.Printf("  Agents: %d | Rounds: %d | Threshold: %.2f\n",
			opts.NumAgents, opts.MaxRounds, opts.ConsensusThreshold)
		fmt.Printf("  Temperature range: [%.2f, %.2f]\n", opts.TempMin, opts.TempMax)
		fmt.Printf("%s\n\n", strings.Repeat("=", 60))
		for _, a := range agents {
			fmt.Printf("  • %s\n", a)
		}
	}

	engine := NewSimulationEngine(client, model,
		WithMaxRounds(opts.MaxRounds),
		WithConsensusThreshold(opts.ConsensusThreshold),
		WithVerbose(opts.Verbose),
	)

	result := engine.Run(question, agents, "", "general")

	if opts.OutputPath != "" {
		if err := SaveResult(result, opts.OutputPath); err != nil {
			return nil, fmt.Errorf("failed to save results: %w", err)
		}
		if opts.Verbose {
			fmt.Printf("\nResults saved to: %s\n", opts.OutputPath)
		}
	}

	return result, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getInt(m map[string]interface{}, key string) int {
	return toInt(m[key])
}

func toInt(v interface{}) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return int(i)
		}
	}
	return 0
}

func getFloat(m map[string]interface{}, key string) float64 {
	if v, ok := m[key].(float64); ok {
		return v
	}
	return 0.0
}

func getBool(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}
