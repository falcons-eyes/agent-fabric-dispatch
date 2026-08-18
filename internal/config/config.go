package config

import (
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// FabricDir returns the path to the fabric configuration directory.
func FabricDir() string {
	return filepath.Join(os.Getenv("HOME"), ".fabric")
}

// PolicyPath returns the path to the policy file.
func PolicyPath() string {
	return filepath.Join(FabricDir(), "policy.yaml")
}

// MeteringPath returns the path to the metering log file.
func MeteringPath() string {
	return filepath.Join(FabricDir(), "metering.jsonl")
}

// Policy represents the fabric dispatch policy configuration.
type Policy struct {
	Rules   []Rule           `yaml:"rules"`
	Budgets map[string]string `yaml:"budgets,omitempty"`
	Workers Workers          `yaml:"workers"`
}

// Rule represents a single policy rule.
type Rule struct {
	Match    map[string]interface{} `yaml:"match,omitempty"`
	Route    string                 `yaml:"route,omitempty"`
	Fallback string                 `yaml:"fallback,omitempty"`
	Default  string                 `yaml:"default,omitempty"`
}

// Workers contains worker endpoint configurations.
type Workers struct {
	Local LocalWorker `yaml:"local"`
	Cloud CloudWorker `yaml:"cloud,omitempty"`
}

// LocalWorker represents a local Ollama/vLLM worker.
type LocalWorker struct {
	Endpoint string   `yaml:"endpoint"`
	Models   []string `yaml:"models"`
}

// CloudWorker represents a remote Agent Fabric worker (Phase 2).
type CloudWorker struct {
	FabricService string `yaml:"fabric_service,omitempty"`
	Zone          string `yaml:"zone,omitempty"`
}

// LoadPolicy reads and parses the policy file.
func LoadPolicy() (*Policy, error) {
	data, err := os.ReadFile(PolicyPath())
	if err != nil {
		return nil, err
	}

	policy := &Policy{}
	if err := yaml.Unmarshal(data, policy); err != nil {
		return nil, err
	}

	return policy, nil
}

// EnsureDir ensures the fabric configuration directory exists.
func EnsureDir() error {
	return os.MkdirAll(FabricDir(), 0o755)
}
