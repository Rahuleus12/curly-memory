package simulator

import (
	"time"
)

// ThinkingStyle represents a cognitive style that biases how an agent reasons.
type ThinkingStyle string

const (
	ThinkingAnalytical   ThinkingStyle = "analytical"
	ThinkingCreative     ThinkingStyle = "creative"
	ThinkingPragmatic    ThinkingStyle = "pragmatic"
	ThinkingEmotional    ThinkingStyle = "emotional"
	ThinkingSkeptical    ThinkingStyle = "skeptical"
	ThinkingOptimistic   ThinkingStyle = "optimistic"
	ThinkingConservative ThinkingStyle = "conservative"
	ThinkingCollaborative ThinkingStyle = "collaborative"
)

// AllThinkingStyles returns all valid thinking styles.
func AllThinkingStyles() []ThinkingStyle {
	return []ThinkingStyle{
		ThinkingAnalytical,
		ThinkingCreative,
		ThinkingPragmatic,
		ThinkingEmotional,
		ThinkingSkeptical,
		ThinkingOptimistic,
		ThinkingConservative,
		ThinkingCollaborative,
	}
}

// EducationLevel represents the highest level of education completed.
type EducationLevel string

const (
	EducationHighSchool EducationLevel = "high_school"
	EducationBachelors EducationLevel = "bachelors"
	EducationMasters   EducationLevel = "masters"
	EducationDoctorate EducationLevel = "doctorate"
	EducationSelfTaught EducationLevel = "self_taught"
)

// AllEducationLevels returns all valid education levels.
func AllEducationLevels() []EducationLevel {
	return []EducationLevel{
		EducationHighSchool,
		EducationBachelors,
		EducationMasters,
		EducationDoctorate,
		EducationSelfTaught,
	}
}

// PersonalityArchetype represents broad personality archetypes that influence agent behavior.
type PersonalityArchetype string

const (
	ArchetypeAnalytical   PersonalityArchetype = "analytical"
	ArchetypeCreative     PersonalityArchetype = "creative"
	ArchetypePragmatic    PersonalityArchetype = "pragmatic"
	ArchetypeEmpathetic   PersonalityArchetype = "empathetic"
	ArchetypeSkeptical    PersonalityArchetype = "skeptical"
	ArchetypeOptimistic   PersonalityArchetype = "optimistic"
	ArchetypeConservative PersonalityArchetype = "conservative"
	ArchetypeProgressive  PersonalityArchetype = "progressive"
	ArchetypeDiplomatic   PersonalityArchetype = "diplomatic"
	ArchetypeDirect       PersonalityArchetype = "direct"
)

// PoliticalLean represents the political leaning of an agent.
type PoliticalLean string

const (
	PoliticalLiberal     PoliticalLean = "liberal"
	PoliticalModerate    PoliticalLean = "moderate"
	PoliticalConservative PoliticalLean = "conservative"
	PoliticalLibertarian PoliticalLean = "libertarian"
	PoliticalApolitical  PoliticalLean = "apolitical"
)

// AgentProfile contains biographical and psychological profile of a simulated person.
type AgentProfile struct {
	Name               string
	Age                int
	Occupation         string
	Education          EducationLevel
	Background         string
	PersonalityTraits  []string
	ThinkingStyle      ThinkingStyle
	Values             []string
	CommunicationStyle string
	Bio                string
}

// AgentConfig contains runtime configuration for an agent.
type AgentConfig struct {
	Temperature      float64
	MaxTokens        int
	TopP             float64
	FrequencyPenalty float64
	PresencePenalty  float64
}

// DefaultAgentConfig returns an AgentConfig with sensible defaults.
func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		Temperature:      0.7,
		MaxTokens:        300,
		TopP:             0.9,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
	}
}

// AgentPersona defines a complete persona for a simulated agent (used in prompts).
type AgentPersona struct {
	Name               string
	Age                int
	Occupation         string
	Background         string
	PersonalityTraits  []string
	CommunicationStyle string
	Biases             []string
	ExpertiseAreas     []string
}

// RoundResponse represents a single agent's response within one round.
type RoundResponse struct {
	AgentName     string  `json:"agent_name"`
	RoundNumber   int     `json:"round_number"`
	Temperature   float64 `json:"temperature"`
	ThinkingStyle string  `json:"thinking_style"`
	Text          string  `json:"text"`
	Stance        string  `json:"stance,omitempty"`
	Timestamp     string  `json:"timestamp"`
}

// NewRoundResponse creates a RoundResponse with the current timestamp if not provided.
func NewRoundResponse(agentName string, roundNumber int, temperature float64, thinkingStyle, text string) RoundResponse {
	if thinkingStyle == "" {
		thinkingStyle = "unknown"
	}
	return RoundResponse{
		AgentName:     agentName,
		RoundNumber:   roundNumber,
		Temperature:   temperature,
		ThinkingStyle: thinkingStyle,
		Text:          text,
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
	}
}

// RoundResult contains aggregated results for one discussion round.
type RoundResult struct {
	RoundNumber        int                       `json:"round_number"`
	Responses          []RoundResponse           `json:"responses"`
	ModeratorSummary   string                    `json:"moderator_summary,omitempty"`
	ConsensusReached   bool                      `json:"consensus_reached"`
	StanceDistribution map[string]int            `json:"stance_distribution"`
}

// NewRoundResult creates an empty RoundResult.
func NewRoundResult(roundNumber int) RoundResult {
	return RoundResult{
		RoundNumber:        roundNumber,
		Responses:          make([]RoundResponse, 0),
		StanceDistribution: make(map[string]int),
	}
}

// SimulationResult contains complete results of a simulation run.
type SimulationResult struct {
	Question         string                 `json:"question"`
	Context          string                 `json:"context"`
	Category         string                 `json:"category"`
	Rounds           []RoundResult          `json:"rounds"`
	FinalConsensus   map[string]interface{} `json:"final_consensus,omitempty"`
	TotalRounds      int                    `json:"total_rounds"`
	ConsensusReached bool                   `json:"consensus_reached"`
	Transcript       string                 `json:"transcript"`
	AgentProfiles    []map[string]interface{} `json:"agent_profiles"`
	StartedAt        string                 `json:"started_at"`
	FinishedAt       string                 `json:"finished_at"`
}

// NewSimulationResult creates an empty SimulationResult.
func NewSimulationResult(question string) SimulationResult {
	return SimulationResult{
		Question:      question,
		Rounds:        make([]RoundResult, 0),
		AgentProfiles: make([]map[string]interface{}, 0),
	}
}

// ModelConfig contains configuration for the LLM model.
type ModelConfig struct {
	APIKey         string `json:"api_key"`
	APIBase        string `json:"api_base"`
	ModelName      string `json:"model_name"`
	MaxTokens      int    `json:"max_tokens"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}

// DefaultModelConfig returns a ModelConfig with sensible defaults.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		APIBase:        "https://api.openai.com/v1",
		ModelName:      "gpt-4",
		MaxTokens:      512,
		TimeoutSeconds: 60,
	}
}

// SimulationSettings contains top-level settings for a simulation run.
type SimulationSettings struct {
	MaxRounds           int     `json:"max_rounds"`
	ConsensusThreshold  float64 `json:"consensus_threshold"`
	ParallelRequests    bool    `json:"parallel_requests"`
	Verbose             bool    `json:"verbose"`
	OutputDir           string  `json:"output_dir"`
	SaveTranscript      bool    `json:"save_transcript"`
	SaveSummary         bool    `json:"save_summary"`
}

// DefaultSimulationSettings returns SimulationSettings with sensible defaults.
func DefaultSimulationSettings() SimulationSettings {
	return SimulationSettings{
		MaxRounds:          5,
		ConsensusThreshold: 0.7,
		ParallelRequests:   true,
		Verbose:            false,
		OutputDir:          "output",
		SaveTranscript:     true,
		SaveSummary:        true,
	}
}

// QuestionConfig represents a question to be debated by the simulated agents.
type QuestionConfig struct {
	Text            string   `json:"text"`
	Context         string   `json:"context"`
	Category        string   `json:"category"`
	ExpectedStances []string `json:"expected_stances"`
}

// BuildQuestionPrompt builds the full question prompt including any context.
func (q *QuestionConfig) BuildQuestionPrompt() string {
	prompt := "Question: " + q.Text
	if q.Context != "" {
		prompt += "\n\nContext: " + q.Context
	}
	return prompt
}

// SimulationConfig is the complete configuration for a simulation run.
type SimulationConfig struct {
	Model     ModelConfig        `json:"model"`
	Settings  SimulationSettings `json:"settings"`
	Agents    []AgentProfile     `json:"agents"`
	Questions []QuestionConfig   `json:"questions"`
}

// DefaultSimulationConfig returns a SimulationConfig with sensible defaults.
func DefaultSimulationConfig() SimulationConfig {
	return SimulationConfig{
		Model:    DefaultModelConfig(),
		Settings: DefaultSimulationSettings(),
		Agents:   make([]AgentProfile, 0),
		Questions: make([]QuestionConfig, 0),
	}
}

// ConfigFileFormat represents the JSON config file structure.
type ConfigFileFormat struct {
	Question            string          `json:"question"`
	Context             string          `json:"context"`
	Category            string          `json:"category"`
	NumAgents           int             `json:"num_agents"`
	MaxRounds           int             `json:"max_rounds"`
	ConsensusThreshold  float64         `json:"consensus_threshold"`
	Model               string          `json:"model"`
	TemperatureMin      float64         `json:"temperature_min"`
	TemperatureMax      float64         `json:"temperature_max"`
	Seed                *int64          `json:"seed"`
	Parallel            bool            `json:"parallel"`
	PersonaIndices      []int           `json:"persona_indices,omitempty"`
	OutputSettings      OutputSettings  `json:"output_settings,omitempty"`
}

// OutputSettings contains settings for output file generation.
type OutputSettings struct {
	OutputDir    string `json:"output_dir"`
	SaveJSON     bool   `json:"save_json"`
	SaveMarkdown bool   `json:"save_markdown"`
	Verbose      bool   `json:"verbose"`
}

// DefaultOutputSettings returns OutputSettings with sensible defaults.
func DefaultOutputSettings() OutputSettings {
	return OutputSettings{
		OutputDir:    "output",
		SaveJSON:     true,
		SaveMarkdown: true,
		Verbose:      true,
	}
}
