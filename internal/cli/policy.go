package cli

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"
)

var policyCmd = &cobra.Command{
	Use:   "policy",
	Short: "Manage dispatch policy rules",
}

var policySetCmd = &cobra.Command{
	Use:   "set <match> <route>",
	Short: "Add or update a policy rule",
	Long: `Add a policy rule. Examples:
  fabric policy set path:secrets/  local-only
  fabric policy set task:refactor_bulk,summarize_files,classify  local
  fabric policy set repo_tag:corp  local-only`,
	Args: cobra.ExactArgs(2),
	RunE: runPolicySet,
}

var policyShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Display the current policy",
	RunE:  runPolicyShow,
}

var policyResetCmd = &cobra.Command{
	Use:   "reset",
	Short: "Reset policy to default",
	RunE:  runPolicyReset,
}

func init() {
	policyCmd.AddCommand(policySetCmd)
	policyCmd.AddCommand(policyShowCmd)
	policyCmd.AddCommand(policyResetCmd)
}

type Policy struct {
	Rules   []Rule           `yaml:"rules"`
	Budgets map[string]string `yaml:"budgets,omitempty"`
	Workers WorkerConfig     `yaml:"workers,omitempty"`
}

type Rule struct {
	Match    map[string]interface{} `yaml:"match,omitempty"`
	Route    string                 `yaml:"route,omitempty"`
	Fallback string                 `yaml:"fallback,omitempty"`
	Default  string                 `yaml:"default,omitempty"`
}

type WorkerConfig struct {
	Local LocalWorker `yaml:"local,omitempty"`
	Cloud CloudWorker `yaml:"cloud,omitempty"`
}

type LocalWorker struct {
	Endpoint string   `yaml:"endpoint"`
	Models   []string `yaml:"models"`
}

type CloudWorker struct {
	FabricService string `yaml:"fabric_service,omitempty"`
	Zone          string `yaml:"zone,omitempty"`
}

func policyPath() string {
	return filepath.Join(os.Getenv("HOME"), ".fabric", "policy.yaml")
}

func loadPolicy() (*Policy, error) {
	path := policyPath()
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return defaultPolicy(), nil
		}
		return nil, fmt.Errorf("failed to read policy: %w", err)
	}

	policy := &Policy{}
	if err := yaml.Unmarshal(data, policy); err != nil {
		return nil, fmt.Errorf("failed to parse policy: %w", err)
	}
	return policy, nil
}

func savePolicy(policy *Policy) error {
	path := policyPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	data, err := yaml.Marshal(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal policy: %w", err)
	}

	return os.WriteFile(path, data, 0o644)
}

func defaultPolicy() *Policy {
	return &Policy{
		Rules: []Rule{
			{Default: "FRONTIER"},
		},
		Budgets: map[string]string{
			"frontier_daily": "5.00",
		},
		Workers: WorkerConfig{
			Local: LocalWorker{
				Endpoint: "http://127.0.0.1:11434",
				Models:   []string{"qwen3-coder:30b"},
			},
		},
	}
}

func runPolicySet(cmd *cobra.Command, args []string) error {
	matchStr := args[0]
	routeStr := strings.ToUpper(strings.ReplaceAll(args[1], "-", "_"))

	policy, err := loadPolicy()
	if err != nil {
		return err
	}

	// Parse match: "path:secrets/" or "task:refactor_bulk,summarize_files"
	parts := strings.SplitN(matchStr, ":", 2)
	if len(parts) != 2 {
		return fmt.Errorf("invalid match format: use key:value (e.g., path:secrets/, task:refactor_bulk)")
	}

	matchKey := parts[0]
	matchValues := strings.Split(parts[1], ",")

	// Build match map
	match := map[string]interface{}{}
	if len(matchValues) == 1 && matchKey != "task" {
		match[matchKey] = []string{matchValues[0]}
	} else {
		match[matchKey] = matchValues
	}

	// Create new rule
	newRule := Rule{
		Match: match,
		Route: routeStr,
	}

	// Add fallback for LOCAL routes
	if routeStr == "LOCAL" {
		newRule.Fallback = "FRONTIER"
	}

	// Insert before the default rule
	inserted := false
	for i, rule := range policy.Rules {
		if rule.Default != "" {
			policy.Rules = append(policy.Rules[:i], append([]Rule{newRule}, policy.Rules[i:]...)...)
			inserted = true
			break
		}
	}
	if !inserted {
		policy.Rules = append(policy.Rules, newRule)
	}

	if err := savePolicy(policy); err != nil {
		return err
	}

	fmt.Printf("Policy rule added: %s → %s\n", matchStr, routeStr)
	return nil
}

func runPolicyShow(cmd *cobra.Command, args []string) error {
	data, err := os.ReadFile(policyPath())
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("No policy configured. Using defaults.")
			fmt.Println("Run 'fabric policy set' to add rules.")
			return nil
		}
		return err
	}

	fmt.Printf("# %s\n", policyPath())
	fmt.Println(string(data))
	return nil
}

func runPolicyReset(cmd *cobra.Command, args []string) error {
	policy := defaultPolicy()
	if err := savePolicy(policy); err != nil {
		return err
	}
	fmt.Println("Policy reset to defaults.")
	return nil
}
