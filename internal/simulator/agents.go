package simulator

import (
	"context"
	"fmt"
	"math/rand"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// ---------------------------------------------------------------------------
// Pool data for generating random agents
// ---------------------------------------------------------------------------

var firstNames = []string{
	"Alice", "Bob", "Carlos", "Diana", "Ethan", "Fatima", "George", "Hannah",
	"Ibrahim", "Julia", "Kevin", "Luna", "Marcus", "Nadia", "Oliver", "Priya",
	"Quinn", "Rachel", "Samuel", "Tara", "Uma", "Victor", "Wendy", "Yuki", "Zara",
}

var occupations = []string{
	"Software Engineer", "Teacher", "Doctor", "Nurse", "Accountant", "Artist",
	"Chef", "Journalist", "Lawyer", "Mechanic", "Social Worker", "Farmer",
	"Pharmacist", "Architect", "Electrician", "Marketing Manager",
	"Research Scientist", "Police Officer", "Librarian", "Entrepreneur",
	"Truck Driver", "Graphic Designer", "Financial Analyst", "Civil Engineer", "Veterinarian",
}

var personalityTraits = []string{
	"outgoing", "reserved", "detail-oriented", "big-picture thinker", "empathetic",
	"stubborn", "open-minded", "cautious", "bold", "patient", "impatient",
	"diplomatic", "direct", "humorous", "serious", "idealistic", "realistic",
	"curious", "loyal", "independent", "team-oriented", "perfectionist", "adaptable",
}

var coreValues = []string{
	"honesty", "fairness", "freedom", "security", "tradition", "innovation",
	"community", "individuality", "sustainability", "efficiency", "compassion",
	"justice", "loyalty", "creativity", "stability", "progress", "equality",
	"meritocracy", "family", "adventure", "knowledge", "spirituality", "health", "wealth",
}

var communicationStyles = []string{
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
}

var backgrounds = []string{
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
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

// Agent represents a simulated person backed by an LLM.
// The agent has a persistent identity (profile) and configuration (temperature, etc.)
// that together shape every response it gives.
type Agent struct {
	Profile AgentProfile
	Config  AgentConfig
	Client  *openai.Client
	Model   string
	history []agentHistoryEntry
}

type agentHistoryEntry struct {
	Round    int    `json:"round"`
	Question string `json:"question"`
	Response string `json:"response"`
	IsFinal  bool   `json:"is_final"`
}

// NewAgent creates a new Agent with the given profile, config, and client.
func NewAgent(profile AgentProfile, config AgentConfig, client *openai.Client, model string) *Agent {
	return &Agent{
		Profile: profile,
		Config:  config,
		Client:  client,
		Model:   model,
		history: make([]agentHistoryEntry, 0),
	}
}

// Name returns the agent's name.
func (a *Agent) Name() string {
	return a.Profile.Name
}

// Temperature returns the agent's temperature setting.
func (a *Agent) Temperature() float64 {
	return a.Config.Temperature
}

// ThinkingStyleValue returns the agent's thinking style as a string.
func (a *Agent) ThinkingStyleValue() ThinkingStyle {
	return a.Profile.ThinkingStyle
}

// BuildSystemPrompt builds a rich system prompt that encodes the agent's identity.
func (a *Agent) BuildSystemPrompt(contextStr string) string {
	var parts []string

	parts = append(parts, fmt.Sprintf(
		"You are %s, a %d-year-old %s.",
		a.Profile.Name,
		a.Profile.Age,
		strings.ToLower(a.Profile.Occupation),
	))

	eduStr := strings.ReplaceAll(string(a.Profile.Education), "_", " ")
	parts = append(parts, fmt.Sprintf("Education: %s.", eduStr))

	parts = append(parts, fmt.Sprintf("Background: %s", a.Profile.Background))
	parts = append(parts, fmt.Sprintf("Biographical details: %s", a.Profile.Bio))

	traits := strings.Join(a.Profile.PersonalityTraits, ", ")
	parts = append(parts, fmt.Sprintf("Personality traits: %s.", traits))

	parts = append(parts, fmt.Sprintf(
		"Thinking style: %s. Respond in a way that reflects this cognitive approach.",
		a.Profile.ThinkingStyle,
	))

	values := strings.Join(a.Profile.Values, ", ")
	parts = append(parts, fmt.Sprintf("Core values: %s.", values))

	parts = append(parts, fmt.Sprintf("Communication style: %s.", a.Profile.CommunicationStyle))

	parts = append(parts, `
IMPORTANT RULES:
1. Stay in character at all times.
2. Express opinions that are consistent with your background, values, and personality.
3. Be authentic — real people are sometimes uncertain, opinionated, or ambivalent.
4. Keep responses concise (2-5 sentences) unless the topic warrants more detail.
5. When responding to others, acknowledge their points before adding your own.
6. You may change your mind if presented with compelling arguments, but only if it fits your character.`)

	if contextStr != "" {
		parts = append(parts, fmt.Sprintf("\nAdditional context for this discussion:\n%s", contextStr))
	}

	return strings.Join(parts, "\n\n")
}

// PreviousResponse represents a previous response in the discussion.
type PreviousResponse struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

// Respond generates a response from this agent.
func (a *Agent) Respond(question string, previousResponses []PreviousResponse, contextStr string, roundNumber int, isFinal bool) string {
	systemPrompt := a.BuildSystemPrompt(contextStr)

	// Build user message
	var userParts []string

	if roundNumber == 1 && len(previousResponses) == 0 {
		userParts = append(userParts, fmt.Sprintf(
			"Discussion Topic: %s\n\nPlease share your initial thoughts and position on this topic.",
			question,
		))
	} else {
		userParts = append(userParts, fmt.Sprintf("Discussion Topic: %s", question))
		userParts = append(userParts, fmt.Sprintf("(Round %d of discussion)\n", roundNumber))

		if len(previousResponses) > 0 {
			userParts = append(userParts, "Here is what others have said so far:\n")
			for _, resp := range previousResponses {
				userParts = append(userParts, fmt.Sprintf("- %s: %s", resp.Name, resp.Text))
			}
			userParts = append(userParts, "")
		}

		if isFinal {
			userParts = append(userParts,
				"This is the FINAL round. Please state your conclusive "+
					"position. Begin your response with either 'AGREE:' or "+
					"'DISAGREE:' followed by a brief summary of your position.")
		} else {
			userParts = append(userParts,
				"Consider what others have said and respond with your "+
					"current thoughts. You may agree, disagree, or propose "+
					"a compromise.")
		}
	}

	userMessage := strings.Join(userParts, "\n")

	// Call the LLM
	messages := []openai.ChatCompletionMessage{
		{Role: openai.ChatMessageRoleSystem, Content: systemPrompt},
		{Role: openai.ChatMessageRoleUser, Content: userMessage},
	}

	resp, err := a.Client.CreateChatCompletion(
		context.Background(),
		a.getCreateParams(messages),
	)

	var text string
	if err != nil {
		text = fmt.Sprintf("[Error generating response: %v]", err)
	} else if len(resp.Choices) > 0 {
		text = resp.Choices[0].Message.Content
	}

	// Record in local history
	a.history = append(a.history, agentHistoryEntry{
		Round:    roundNumber,
		Question: question,
		Response: text,
		IsFinal:  isFinal,
	})

	return strings.TrimSpace(text)
}

// getCreateParams creates the chat completion request parameters.
func (a *Agent) getCreateParams(messages []openai.ChatCompletionMessage) openai.ChatCompletionRequest {
	return openai.ChatCompletionRequest{
		Model:            a.Model,
		Messages:         messages,
		Temperature:      float32(a.Config.Temperature),
		MaxTokens:        a.Config.MaxTokens,
		TopP:             float32(a.Config.TopP),
		FrequencyPenalty: float32(a.Config.FrequencyPenalty),
		PresencePenalty:  float32(a.Config.PresencePenalty),
	}
}

// ExtractStance tries to extract a clear AGREE / DISAGREE stance from a final response.
// Returns "agree", "disagree", or "" if unclear.
func (a *Agent) ExtractStance(text string) string {
	upper := strings.ToUpper(strings.TrimSpace(text))
	if strings.HasPrefix(upper, "AGREE") {
		return "agree"
	}
	if strings.HasPrefix(upper, "DISAGREE") {
		return "disagree"
	}

	// Fallback: look for keywords
	agreeKeywords := []string{
		"i agree", "i support", "i'm in favor", "i'm for",
		"consensus", "common ground", "compromise",
	}
	disagreeKeywords := []string{
		"i disagree", "i oppose", "i'm against", "i reject",
		"cannot agree", "do not support",
	}

	lower := strings.ToLower(text)
	for _, kw := range agreeKeywords {
		if strings.Contains(lower, kw) {
			return "agree"
		}
	}
	for _, kw := range disagreeKeywords {
		if strings.Contains(lower, kw) {
			return "disagree"
		}
	}

	return ""
}

// String returns a string representation of the agent.
func (a *Agent) String() string {
	return fmt.Sprintf("%s (%d, %s) — thinking: %s, temp: %.2f",
		a.Profile.Name,
		a.Profile.Age,
		a.Profile.Occupation,
		a.Profile.ThinkingStyle,
		a.Config.Temperature,
	)
}

// ---------------------------------------------------------------------------
// Agent Factory
// ---------------------------------------------------------------------------

// AgentFactory creates diverse groups of simulated agents.
// Each agent is given a unique profile and a temperature setting
// that together produce varied, realistic responses.
type AgentFactory struct {
	Client     *openai.Client
	Model      string
	BaseConfig AgentConfig
	rng        *rand.Rand
}

// NewAgentFactory creates a new AgentFactory.
// If seed is negative, a random seed will be used.
func NewAgentFactory(client *openai.Client, model string, baseConfig *AgentConfig, seed int64) *AgentFactory {
	if baseConfig == nil {
		defaultConfig := DefaultAgentConfig()
		baseConfig = &defaultConfig
	}

	var rng *rand.Rand
	if seed < 0 {
		rng = rand.New(rand.NewSource(rand.Int63()))
	} else {
		rng = rand.New(rand.NewSource(seed))
	}

	return &AgentFactory{
		Client:     client,
		Model:      model,
		BaseConfig: *baseConfig,
		rng:        rng,
	}
}

// pick selects count unique items from pool.
func (f *AgentFactory) pick(pool []string, count int) []string {
	if count >= len(pool) {
		// Return a shuffled copy
		result := make([]string, len(pool))
		copy(result, pool)
		f.rng.Shuffle(len(result), func(i, j int) {
			result[i], result[j] = result[j], result[i]
		})
		return result
	}

	indices := f.rng.Perm(len(pool))
	result := make([]string, count)
	for i := 0; i < count; i++ {
		result[i] = pool[indices[i]]
	}
	return result
}

// pickThinkingStyle selects count unique thinking styles from all available.
func (f *AgentFactory) pickThinkingStyle(count int) []ThinkingStyle {
	all := AllThinkingStyles()
	if count >= len(all) {
		result := make([]ThinkingStyle, len(all))
		copy(result, all)
		f.rng.Shuffle(len(result), func(i, j int) {
			result[i], result[j] = result[j], result[i]
		})
		return result
	}

	indices := f.rng.Perm(len(all))
	result := make([]ThinkingStyle, count)
	for i := 0; i < count; i++ {
		result[i] = all[indices[i]]
	}
	return result
}

// pickEducationLevel selects a random education level.
func (f *AgentFactory) pickEducationLevel() EducationLevel {
	all := AllEducationLevels()
	return all[f.rng.Intn(len(all))]
}

// generateBio creates a short biographical blurb from the profile.
func (f *AgentFactory) generateBio(profile *AgentProfile) string {
	edu := strings.ReplaceAll(string(profile.Education), "_", " ")
	traits := strings.Join(profile.PersonalityTraits[:safeSliceLen(len(profile.PersonalityTraits), 3)], ", ")
	return fmt.Sprintf(
		"A %d-year-old %s with a %s background. Known for being %s. %s Approaches problems with a %s mindset.",
		profile.Age,
		strings.ToLower(profile.Occupation),
		edu,
		traits,
		profile.Background,
		profile.ThinkingStyle,
	)
}

// safeSliceLen returns min(a, b) for slice length calculations.
func safeSliceLen(length, max int) int {
	if length < max {
		return length
	}
	return max
}

// CreateAgentOptions contains optional parameters for creating a single agent.
type CreateAgentOptions struct {
	Name               *string
	Age                *int
	Occupation         *string
	Temperature        *float64
	ThinkingStyle      *ThinkingStyle
	Education          *EducationLevel
	Background         *string
	PersonalityTraits  []string
	Values             []string
	CommunicationStyle *string
}

// CreateAgent creates a single agent with the given or random attributes.
func (f *AgentFactory) CreateAgent(opts CreateAgentOptions) *Agent {
	// Resolve name
	name := ""
	if opts.Name != nil {
		name = *opts.Name
	} else {
		name = firstNames[f.rng.Intn(len(firstNames))]
	}

	// Resolve age
	age := 22 + f.rng.Intn(47) // 22-68
	if opts.Age != nil {
		age = *opts.Age
	}

	// Resolve occupation
	occupation := occupations[f.rng.Intn(len(occupations))]
	if opts.Occupation != nil {
		occupation = *opts.Occupation
	}

	// Resolve education
	education := f.pickEducationLevel()
	if opts.Education != nil {
		education = *opts.Education
	}

	// Resolve thinking style
	thinking := AllThinkingStyles()[f.rng.Intn(len(AllThinkingStyles()))]
	if opts.ThinkingStyle != nil {
		thinking = *opts.ThinkingStyle
	}

	// Resolve personality traits
	traits := opts.PersonalityTraits
	if len(traits) == 0 {
		count := 2 + f.rng.Intn(3) // 2-4 traits
		traits = f.pick(personalityTraits, count)
	}

	// Resolve values
	values := opts.Values
	if len(values) == 0 {
		count := 2 + f.rng.Intn(3) // 2-4 values
		values = f.pick(coreValues, count)
	}

	// Resolve background
	bg := backgrounds[f.rng.Intn(len(backgrounds))]
	if opts.Background != nil {
		bg = *opts.Background
	}

	// Resolve communication style
	comm := communicationStyles[f.rng.Intn(len(communicationStyles))]
	if opts.CommunicationStyle != nil {
		comm = *opts.CommunicationStyle
	}

	profile := AgentProfile{
		Name:               name,
		Age:                age,
		Occupation:         occupation,
		Education:          education,
		Background:         bg,
		PersonalityTraits:  traits,
		ThinkingStyle:      thinking,
		Values:             values,
		CommunicationStyle: comm,
		Bio:                "", // filled below
	}
	profile.Bio = f.generateBio(&profile)

	// Resolve temperature
	temp := f.BaseConfig.Temperature
	if opts.Temperature != nil {
		temp = *opts.Temperature
	}

	config := AgentConfig{
		Temperature:      temp,
		MaxTokens:        f.BaseConfig.MaxTokens,
		TopP:             f.BaseConfig.TopP,
		FrequencyPenalty: f.BaseConfig.FrequencyPenalty,
		PresencePenalty:  f.BaseConfig.PresencePenalty,
	}

	return NewAgent(profile, config, f.Client, f.Model)
}

// CreateGroup creates a diverse group of agents.
// Temperatures are spread evenly across the temperature range.
// If ensureDiversity is true, each agent gets a unique name, occupation, and thinking style as far as the pools allow.
func (f *AgentFactory) CreateGroup(count int, tempMin, tempMax float64, ensureDiversity bool) []*Agent {
	agents := make([]*Agent, 0, count)

	usedNames := make(map[string]bool)
	usedOccupations := make(map[string]bool)
	usedThinking := make(map[ThinkingStyle]bool)

	for i := 0; i < count; i++ {
		// Spread temperatures evenly across the range
		var temp float64
		if count > 1 {
			frac := float64(i) / float64(count-1)
			temp = tempMin + frac*(tempMax-tempMin)
		} else {
			temp = (tempMin + tempMax) / 2
		}
		// Round to 2 decimal places
		temp = float64(int(temp*100)) / 100

		var name, occupation *string
		var thinking *ThinkingStyle

		if ensureDiversity {
			// Pick unique name
			for _, n := range firstNames {
				if !usedNames[n] {
					name = &n
					usedNames[n] = true
					break
				}
			}

			// Pick unique occupation
			for _, o := range occupations {
				if !usedOccupations[o] {
					occupation = &o
					usedOccupations[o] = true
					break
				}
			}

			// Pick unique thinking style
			for _, t := range AllThinkingStyles() {
				if !usedThinking[t] {
					th := t
					thinking = &th
					usedThinking[t] = true
					break
				}
			}
		}

		agent := f.CreateAgent(CreateAgentOptions{
			Name:          name,
			Occupation:    occupation,
			Temperature:   &temp,
			ThinkingStyle: thinking,
		})
		agents = append(agents, agent)
	}

	return agents
}

// CreateCustomGroup creates agents from a list of specification options.
// Missing fields in each spec are filled with random values.
func (f *AgentFactory) CreateCustomGroup(specs []CreateAgentOptions) []*Agent {
	agents := make([]*Agent, 0, len(specs))
	for _, spec := range specs {
		agent := f.CreateAgent(spec)
		agents = append(agents, agent)
	}
	return agents
}

// CreateGroupFromPersonas creates agents from pre-defined persona templates.
func (f *AgentFactory) CreateGroupFromPersonas(personas []AgentPersona, tempMin, tempMax float64) []*Agent {
	agents := make([]*Agent, 0, len(personas))

	for i, persona := range personas {
		var temp float64
		if len(personas) > 1 {
			frac := float64(i) / float64(len(personas)-1)
			temp = tempMin + frac*(tempMax-tempMin)
		} else {
			temp = (tempMin + tempMax) / 2
		}
		temp = float64(int(temp*100)) / 100

		profile := AgentProfile{
			Name:               persona.Name,
			Age:                persona.Age,
			Occupation:         persona.Occupation,
			Background:         persona.Background,
			PersonalityTraits:  persona.PersonalityTraits,
			Values:             persona.Biases, // Use biases as values
			CommunicationStyle: persona.CommunicationStyle,
			ThinkingStyle:      ThinkingPragmatic,  // Default
			Education:          EducationBachelors, // Default
		}

		config := AgentConfig{
			Temperature:      temp,
			MaxTokens:        f.BaseConfig.MaxTokens,
			TopP:             f.BaseConfig.TopP,
			FrequencyPenalty: f.BaseConfig.FrequencyPenalty,
			PresencePenalty:  f.BaseConfig.PresencePenalty,
		}

		agent := NewAgent(profile, config, f.Client, f.Model)
		agents = append(agents, agent)
	}

	return agents
}
