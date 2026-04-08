package simulator

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

// UTCTimeFormat is the default format string for timestamps.
const UTCTimeFormat = "20060102_150405"

// TimestampStr returns a filename-safe timestamp string.
func TimestampStr() string {
	return time.Now().UTC().Format(UTCTimeFormat)
}

// UTCTimeStr returns an ISO-format UTC timestamp string.
func UTCTimeStr() string {
	return time.Now().UTC().Format(time.RFC3339)
}

// SafeFilename converts an arbitrary string into a safe filename component.
// Non-alphanumeric characters are replaced with underscores and
// consecutive underscores are collapsed.
func SafeFilename(name string) string {
	// Replace non-alphanumeric characters (except underscore and hyphen) with underscore
	reg := regexp.MustCompile(`[^A-Za-z0-9_\-]`)
	safe := reg.ReplaceAllString(name, "_")
	// Collapse consecutive underscores
	multiUnderscore := regexp.MustCompile(`_+`)
	safe = multiUnderscore.ReplaceAllString(safe, "_")
	safe = strings.Trim(safe, "_")
	return strings.ToLower(safe)
}

// ---------------------------------------------------------------------------
// Stance extraction
// ---------------------------------------------------------------------------

var stanceTagRegex = regexp.MustCompile(`(?i)<stance>\s*(.+?)\s*</stance>`)

// ExtractStanceTag extracts the content of a <stance>...</stance> tag from text.
// Returns the stripped tag content, or empty string if no tag is found.
func ExtractStanceTag(text string) string {
	match := stanceTagRegex.FindStringSubmatch(text)
	if len(match) > 1 {
		return strings.TrimSpace(match[1])
	}
	return ""
}

// Strongly for keywords
var stronglyForKeywords = []string{
	"strongly support", "strongly favor", "strongly agree", "strongly for",
	"fully support", "enthusiastically", "wholeheartedly",
	"absolutely support", "definitely support", "strongly in favor",
}

// Strongly against keywords
var stronglyAgainstKeywords = []string{
	"strongly oppose", "strongly against", "strongly disagree",
	"firmly against", "categorically oppose", "absolutely oppose", "strongly object",
}

// Somewhat for keywords
var somewhatForKeywords = []string{
	"i support", "i agree", "i'm in favor", "i favor", "i'm for",
	"generally support", "lean toward supporting", "tend to agree",
	"mostly agree", "i'm inclined to support", "i believe we should",
}

// Somewhat against keywords
var somewhatAgainstKeywords = []string{
	"i oppose", "i disagree", "i'm against", "i'm not in favor",
	"lean toward opposing", "tend to disagree", "mostly disagree",
	"cannot support", "do not support",
}

// Neutral keywords
var neutralKeywords = []string{
	"neutral", "undecided", "on the fence", "mixed feelings",
	"ambivalent", "can see both sides", "neither for nor against",
	"no strong opinion", "i'm torn", "balance of",
	"need more information", "need more data",
}

// ClassifyStance classifies an agent's stance as one of: strongly_for, somewhat_for,
// neutral, somewhat_against, strongly_against, or unclear.
// The function first looks for a <stance> tag. If none is found
// it falls back to keyword heuristics on the full text.
func ClassifyStance(text string) string {
	stanceText := ExtractStanceTag(text)
	if stanceText == "" {
		stanceText = text
	}

	lower := strings.ToLower(stanceText)

	// Strong signals
	for _, kw := range stronglyForKeywords {
		if strings.Contains(lower, kw) {
			return "strongly_for"
		}
	}
	for _, kw := range stronglyAgainstKeywords {
		if strings.Contains(lower, kw) {
			return "strongly_against"
		}
	}

	// Moderate signals
	for _, kw := range somewhatForKeywords {
		if strings.Contains(lower, kw) {
			return "somewhat_for"
		}
	}
	for _, kw := range somewhatAgainstKeywords {
		if strings.Contains(lower, kw) {
			return "somewhat_against"
		}
	}

	// Neutral signals
	for _, kw := range neutralKeywords {
		if strings.Contains(lower, kw) {
			return "neutral"
		}
	}

	return "unclear"
}

// ---------------------------------------------------------------------------
// Stance extraction for simulation (from simulation.py)
// ---------------------------------------------------------------------------

var stanceRE = regexp.MustCompile(`(?is)<stance>(.*?)</stance>`)

// Agree keywords
var agreeKeywords = []string{
	"strongly support", "strongly favor", "strongly agree",
	"i support", "i favor", "i agree", "i'm in favor", "i'm for",
	"fully support", "definitely support", "absolutely support",
}

// Disagree keywords
var disagreeKeywords = []string{
	"strongly oppose", "strongly against", "strongly disagree",
	"i oppose", "i'm against", "i disagree",
	"cannot support", "do not support", "firmly against",
}

// Neutral stance keywords
var neutralStanceKeywords = []string{
	"neutral", "undecided", "on the fence", "mixed feelings", "ambivalent", "somewhat neutral",
}

// ExtractStance extracts a normalised stance from an agent's response.
// Checks <stance> tags first, then falls back to keyword matching.
// Returns one of: "strongly for", "somewhat for", "neutral",
// "somewhat against", "strongly against", or "unclear".
func ExtractStance(text string) string {
	// Try <stance> tags
	match := stanceRE.FindStringSubmatch(text)
	if len(match) > 1 {
		stanceRaw := strings.TrimSpace(strings.ToLower(match[1]))
		return classifyStanceText(stanceRaw)
	}

	// Fall back to keyword search in the whole text
	lower := strings.ToLower(text)

	for _, kw := range agreeKeywords {
		if strings.Contains(lower, kw) {
			if strings.Contains(lower, "strongly") || strings.Contains(lower, "firmly") {
				return "strongly for"
			}
			return "somewhat for"
		}
	}

	for _, kw := range disagreeKeywords {
		if strings.Contains(lower, kw) {
			if strings.Contains(lower, "strongly") || strings.Contains(lower, "firmly") {
				return "strongly against"
			}
			return "somewhat against"
		}
	}

	for _, kw := range neutralStanceKeywords {
		if strings.Contains(lower, kw) {
			return "neutral"
		}
	}

	return "unclear"
}

// classifyStanceText maps a free-text stance to one of our canonical buckets.
func classifyStanceText(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	text = strings.TrimSuffix(text, ".")

	stronglyForPhrases := []string{"strongly support", "strongly favor", "strongly agree", "fully support"}
	for _, phrase := range stronglyForPhrases {
		if strings.Contains(text, phrase) {
			return "strongly for"
		}
	}

	somewhatForPhrases := []string{"support", "favor", "agree", "for", "in favor", "pro"}
	for _, phrase := range somewhatForPhrases {
		if strings.Contains(text, phrase) {
			return "somewhat for"
		}
	}

	stronglyAgainstPhrases := []string{"strongly oppose", "strongly against", "strongly disagree", "firmly against"}
	for _, phrase := range stronglyAgainstPhrases {
		if strings.Contains(text, phrase) {
			return "strongly against"
		}
	}

	somewhatAgainstPhrases := []string{"oppose", "against", "disagree", "anti", "reject"}
	for _, phrase := range somewhatAgainstPhrases {
		if strings.Contains(text, phrase) {
			return "somewhat against"
		}
	}

	neutralPhrases := []string{"neutral", "undecided", "mixed", "ambivalent", "on the fence"}
	for _, phrase := range neutralPhrases {
		if strings.Contains(text, phrase) {
			return "neutral"
		}
	}

	leanForPhrases := []string{"lean toward", "somewhat support", "mildly support", "tend to agree"}
	for _, phrase := range leanForPhrases {
		if strings.Contains(text, phrase) {
			return "somewhat for"
		}
	}

	leanAgainstPhrases := []string{"lean against", "somewhat oppose", "mildly oppose", "tend to disagree"}
	for _, phrase := range leanAgainstPhrases {
		if strings.Contains(text, phrase) {
			return "somewhat against"
		}
	}

	return "unclear"
}

// ---------------------------------------------------------------------------
// Text formatting
// ---------------------------------------------------------------------------

// WrapText wraps text to width characters per line.
func WrapText(text string, width int) string {
	if width <= 0 {
		width = 80
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return ""
	}

	var lines []string
	currentLine := ""
	currentLen := 0

	for _, word := range words {
		if currentLen == 0 {
			currentLine = word
			currentLen = utf8.RuneCountInString(word)
		} else if currentLen+1+utf8.RuneCountInString(word) <= width {
			currentLine += " " + word
			currentLen += 1 + utf8.RuneCountInString(word)
		} else {
			lines = append(lines, currentLine)
			currentLine = word
			currentLen = utf8.RuneCountInString(word)
		}
	}
	if currentLine != "" {
		lines = append(lines, currentLine)
	}

	return strings.Join(lines, "\n")
}

// IndentText indents every line of text by indent spaces.
func IndentText(text string, indent int) string {
	prefix := strings.Repeat(" ", indent)
	var result []string
	for _, line := range strings.Split(text, "\n") {
		result = append(result, prefix+line)
	}
	return strings.Join(result, "\n")
}

// FormatAgentResponse formats a single agent response for display.
func FormatAgentResponse(name, response string, stance string) string {
	lines := make([]string, 0)
	header := fmt.Sprintf("┌─ %s", name)
	if stance != "" {
		header += fmt.Sprintf("  [%s]", stance)
	}
	lines = append(lines, header)
	lines = append(lines, "│")

	for _, paragraph := range strings.Split(response, "\n") {
		wrapped := WrapText(paragraph, 76)
		for _, line := range strings.Split(wrapped, "\n") {
			lines = append(lines, fmt.Sprintf("│  %s", line))
		}
		lines = append(lines, "│")
	}

	lines = append(lines, "└"+strings.Repeat("─", 77))
	return strings.Join(lines, "\n")
}

// FormatRoundHeader returns a visual header for a discussion round.
func FormatRoundHeader(roundNumber, maxRounds int) string {
	width := 60
	title := fmt.Sprintf("  ROUND %d / %d  ", roundNumber, maxRounds)
	pad := width - len(title)
	left := pad / 2
	right := pad - left
	line := strings.Repeat("═", left) + title + strings.Repeat("═", right)
	return fmt.Sprintf("\n╔%s╗\n║%s║\n╚%s╝", line, strings.Repeat(" ", width), strings.Repeat("═", width))
}

// FormatDivider returns a horizontal divider line.
func FormatDivider(char string, width int) string {
	if char == "" {
		char = "─"
	}
	if width <= 0 {
		width = 60
	}
	return strings.Repeat(char, width)
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// EnsureDir creates path (and parents) if it doesn't exist.
func EnsureDir(path string) error {
	return os.MkdirAll(path, 0755)
}

// SaveJSON saves data as JSON to path.
// Parent directories are created if needed.
func SaveJSON(data interface{}, path string) error {
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	return encoder.Encode(data)
}

// LoadJSON loads and returns JSON data from path.
// The result is decoded into the provided interface.
func LoadJSON(path string, v interface{}) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	return json.NewDecoder(file).Decode(v)
}

// SaveMarkdown writes content as a Markdown file to path.
func SaveMarkdown(content, path string) error {
	if err := EnsureDir(filepath.Dir(path)); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0644)
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

// TranscriptRound represents a round for markdown transcript generation.
type TranscriptRound struct {
	RoundNumber int                    `json:"round_number"`
	Responses   []TranscriptResponse   `json:"responses"`
}

// TranscriptResponse represents a response for markdown transcript generation.
type TranscriptResponse struct {
	Name   string `json:"name"`
	Text   string `json:"text"`
	Stance string `json:"stance,omitempty"`
}

// BuildTranscriptMarkdown builds a full Markdown transcript of a simulation.
func BuildTranscriptMarkdown(question string, rounds []TranscriptRound, summary map[string]interface{}) string {
	var parts []string

	parts = append(parts, "# Consensus Simulation Transcript\n")
	parts = append(parts, fmt.Sprintf("**Generated:** %s\n", UTCTimeStr()))
	parts = append(parts, fmt.Sprintf("## Question\n\n%s\n", question))

	// Agent roster from round 1
	if len(rounds) > 0 && len(rounds[0].Responses) > 0 {
		parts = append(parts, "## Participants\n")
		for _, resp := range rounds[0].Responses {
			parts = append(parts, fmt.Sprintf("- **%s**", resp.Name))
		}
		parts = append(parts, "")
	}

	// Rounds
	for _, rnd := range rounds {
		parts = append(parts, fmt.Sprintf("## Round %d\n", rnd.RoundNumber))
		for _, resp := range rnd.Responses {
			parts = append(parts, fmt.Sprintf("### %s\n", resp.Name))
			if resp.Stance != "" {
				parts = append(parts, fmt.Sprintf("*Stance: %s*\n", resp.Stance))
			}
			parts = append(parts, fmt.Sprintf("%s\n", resp.Text))
			parts = append(parts, "---\n")
		}
	}

	// Summary
	if summary != nil {
		parts = append(parts, "## Summary\n")
		cl, _ := summary["consensus_level"].(string)
		if cl == "" {
			cl = "N/A"
		}
		parts = append(parts, fmt.Sprintf("**Consensus Level:** %s / 5\n", cl))

		if agreement, ok := summary["areas_of_agreement"]; ok {
			parts = append(parts, "### Areas of Agreement\n")
			if list, ok := agreement.([]interface{}); ok {
				for _, item := range list {
					parts = append(parts, fmt.Sprintf("- %v", item))
				}
			} else {
				parts = append(parts, fmt.Sprintf("%v", agreement))
			}
			parts = append(parts, "")
		}

		if disagreement, ok := summary["areas_of_disagreement"]; ok {
			parts = append(parts, "### Areas of Disagreement\n")
			if list, ok := disagreement.([]interface{}); ok {
				for _, item := range list {
					parts = append(parts, fmt.Sprintf("- %v", item))
				}
			} else {
				parts = append(parts, fmt.Sprintf("%v", disagreement))
			}
			parts = append(parts, "")
		}

		if cs, ok := summary["consensus_statement"].(string); ok && cs != "" {
			parts = append(parts, "### Consensus Statement\n")
			parts = append(parts, fmt.Sprintf("%s\n", cs))
		}

		if kp, ok := summary["key_perspectives"]; ok {
			parts = append(parts, "### Key Perspectives\n")
			if dict, ok := kp.(map[string]interface{}); ok {
				for pname, pval := range dict {
					parts = append(parts, fmt.Sprintf("- **%s:** %v", pname, pval))
				}
			} else if list, ok := kp.([]interface{}); ok {
				for _, item := range list {
					parts = append(parts, fmt.Sprintf("- %v", item))
				}
			}
			parts = append(parts, "")
		}
	}

	return strings.Join(parts, "\n")
}

// AgentSummary represents an agent's stance summary.
type AgentSummary struct {
	Name           string `json:"name"`
	InitialStance  string `json:"initial_stance"`
	FinalStance    string `json:"final_stance"`
	StanceChanged  bool   `json:"stance_changed"`
}

// BuildSummaryMarkdown builds a shorter summary report (not the full transcript).
func BuildSummaryMarkdown(question string, totalRounds int, agentSummaries []AgentSummary, consensusInfo map[string]interface{}) string {
	var parts []string

	parts = append(parts, "# Consensus Simulation Summary\n")
	parts = append(parts, fmt.Sprintf("**Generated:** %s\n", UTCTimeStr()))
	parts = append(parts, fmt.Sprintf("**Question:** %s\n", question))
	parts = append(parts, fmt.Sprintf("**Rounds:** %d\n", totalRounds))

	// Stance table
	parts = append(parts, "## Agent Stances\n")
	parts = append(parts, "| Agent | Initial Stance | Final Stance | Changed? |")
	parts = append(parts, "|-------|---------------|-------------|----------|")
	for _, a := range agentSummaries {
		changed := "—"
		if a.StanceChanged {
			changed = "✓"
		}
		initial := a.InitialStance
		if initial == "" {
			initial = "?"
		}
		final := a.FinalStance
		if final == "" {
			final = "?"
		}
		parts = append(parts, fmt.Sprintf("| %s | %s | %s | %s |", a.Name, initial, final, changed))
	}
	parts = append(parts, "")

	// Stance distribution
	if len(agentSummaries) > 0 {
		parts = append(parts, "## Final Stance Distribution\n")
		stanceCounts := make(map[string]int)
		for _, a := range agentSummaries {
			fs := a.FinalStance
			if fs == "" {
				fs = "unclear"
			}
			stanceCounts[fs]++
		}

		// Sort by count (descending)
		type stanceCount struct {
			stance string
			count  int
		}
		var sorted []stanceCount
		for s, c := range stanceCounts {
			sorted = append(sorted, stanceCount{s, c})
		}
		// Simple bubble sort for small slices
		for i := 0; i < len(sorted); i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[j].count > sorted[i].count {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}

		for _, sc := range sorted {
			bar := strings.Repeat("█", sc.count*3)
			parts = append(parts, fmt.Sprintf("- **%s:** %d  %s", sc.stance, sc.count, bar))
		}
		parts = append(parts, "")
	}

	// Consensus info
	if consensusInfo != nil {
		parts = append(parts, "## Consensus Result\n")
		cl, _ := consensusInfo["consensus_level"].(string)
		if cl == "" {
			cl = "N/A"
		}
		parts = append(parts, fmt.Sprintf("- **Consensus Level:** %s / 5\n", cl))

		if cs, ok := consensusInfo["consensus_statement"].(string); ok && cs != "" {
			parts = append(parts, fmt.Sprintf("- **Consensus Statement:** %s\n", cs))
		}
	}

	return strings.Join(parts, "\n")
}

// ---------------------------------------------------------------------------
// Consensus helpers
// ---------------------------------------------------------------------------

// CalculateConsensusLevel calculates a simple consensus score based on stance distribution.
// The score is the fraction of agents that hold the most common stance.
// Returns a value between 0.0 and 1.0.
func CalculateConsensusLevel(stances []string) float64 {
	if len(stances) == 0 {
		return 0.0
	}

	counts := make(map[string]int)
	for _, s := range stances {
		counts[s]++
	}

	maxCount := 0
	for _, count := range counts {
		if count > maxCount {
			maxCount = count
		}
	}

	return float64(maxCount) / float64(len(stances))
}

// GroupStances returns a map of stance label to count.
func GroupStances(stances []string) map[string]int {
	counts := make(map[string]int)
	for _, s := range stances {
		counts[s]++
	}
	return counts
}

// StanceToNumeric maps a stance label to a numeric value.
//   strongly_for → 2.0
//   somewhat_for → 1.0
//   neutral      → 0.0
//   somewhat_against → -1.0
//   strongly_against → -2.0
//   unclear      → 0.0
func StanceToNumeric(stance string) float64 {
	switch stance {
	case "strongly_for":
		return 2.0
	case "somewhat_for":
		return 1.0
	case "neutral":
		return 0.0
	case "somewhat_against":
		return -1.0
	case "strongly_against":
		return -2.0
	default:
		return 0.0
	}
}

// CalculateAverageSentiment calculates the average numeric sentiment across all agents.
// Positive values indicate overall support; negative values indicate overall opposition.
func CalculateAverageSentiment(stances []string) float64 {
	if len(stances) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, s := range stances {
		sum += StanceToNumeric(s)
	}
	return sum / float64(len(stances))
}

// ---------------------------------------------------------------------------
// Stance distribution helpers (from simulation.py)
// ---------------------------------------------------------------------------

// ComputeStanceDistribution counts how many agents fall into each stance bucket.
func ComputeStanceDistribution(responses []RoundResponse) map[string]int {
	dist := make(map[string]int)
	for _, resp := range responses {
		stance := resp.Stance
		if stance == "" {
			stance = "unclear"
		}
		dist[stance]++
	}
	return dist
}

// CheckConsensus determines whether the group has reached consensus.
// Returns (reached, dominantStance).
// threshold is the fraction of agents that must share a position.
func CheckConsensus(responses []RoundResponse, threshold float64) (bool, string) {
	if len(responses) == 0 {
		return false, "none"
	}

	dist := ComputeStanceDistribution(responses)
	total := len(responses)

	// Group "strongly for" + "somewhat for" and similarly for "against"
	grouped := map[string]int{
		"for":     0,
		"against": 0,
		"neutral": 0,
		"unclear": 0,
	}

	for stance, count := range dist {
		if strings.Contains(stance, "for") {
			grouped["for"] += count
		} else if strings.Contains(stance, "against") {
			grouped["against"] += count
		} else if stance == "neutral" {
			grouped["neutral"] += count
		} else {
			grouped["unclear"] += count
		}
	}

	for position, count := range grouped {
		if float64(count)/float64(total) >= threshold {
			return true, position
		}
	}

	return false, "none"
}

// ---------------------------------------------------------------------------
// Transcript helper (from simulation.py)
// ---------------------------------------------------------------------------

// BuildTranscript renders all rounds into a readable transcript string.
func BuildTranscript(rounds []RoundResult) string {
	var parts []string

	for _, rnd := range rounds {
		parts = append(parts, fmt.Sprintf("\n%s", strings.Repeat("=", 60)))
		parts = append(parts, fmt.Sprintf("  ROUND %d", rnd.RoundNumber))
		parts = append(parts, fmt.Sprintf("%s\n", strings.Repeat("=", 60)))

		for _, resp := range rnd.Responses {
			parts = append(parts, fmt.Sprintf("[%s] (temp=%.2f, style=%s)", resp.AgentName, resp.Temperature, resp.ThinkingStyle))
			parts = append(parts, resp.Text)
			if resp.Stance != "" {
				parts = append(parts, fmt.Sprintf("  → Stance: %s", resp.Stance))
			}
			parts = append(parts, "")
		}

		if rnd.ModeratorSummary != "" {
			parts = append(parts, "--- Moderator Summary ---")
			parts = append(parts, rnd.ModeratorSummary)
			parts = append(parts, "")
		}
	}

	return strings.Join(parts, "\n")
}
