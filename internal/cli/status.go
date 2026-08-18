package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show overall fabric dispatch status",
	RunE:  runStatus,
}

func runStatus(cmd *cobra.Command, args []string) error {
	fmt.Println("=== Agent Fabric Dispatch Status ===")
	fmt.Println()

	// Policy status
	pPath := policyPath()
	if _, err := os.Stat(pPath); err == nil {
		fmt.Printf("Policy:   %s\n", pPath)
	} else {
		fmt.Println("Policy:   not configured (using defaults)")
	}

	// Metering status
	meteringPath := filepath.Join(os.Getenv("HOME"), ".fabric", "metering.jsonl")
	if info, err := os.Stat(meteringPath); err == nil {
		lineCount := countLines(meteringPath)
		fmt.Printf("Metering: %s (%d entries, %d bytes)\n", meteringPath, lineCount, info.Size())
	} else {
		fmt.Println("Metering: no data yet")
	}

	// Hooks status
	settingsPath := filepath.Join(os.Getenv("HOME"), ".claude", "settings.json")
	if data, err := os.ReadFile(settingsPath); err == nil {
		var settings ClaudeSettings
		if err := json.Unmarshal(data, &settings); err == nil && len(settings.Hooks) > 0 {
			fmt.Println("Hooks:    installed")
			for name := range settings.Hooks {
				fmt.Printf("          - %s\n", name)
			}
		} else {
			fmt.Println("Hooks:    not installed")
		}
	} else {
		fmt.Println("Hooks:    not installed")
	}

	fmt.Println()
	return nil
}

func countLines(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return 0
	}
	return strings.Count(content, "\n") + 1
}
