package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

var installHooksCmd = &cobra.Command{
	Use:   "install-hooks",
	Short: "Register PreToolUse/PostToolUse/Stop hooks in Claude Code settings",
	RunE:  runInstallHooks,
}

var installMCPCmd = &cobra.Command{
	Use:   "install-mcp",
	Short: "Register fabric-worker as an MCP server in Claude Code",
	RunE:  runInstallMCP,
}

// ClaudeSettings matches the actual ~/.claude/settings.json structure.
type ClaudeSettings struct {
	Hooks      map[string][]HookGroup `json:"hooks,omitempty"`
	McpServers map[string]MCPServer   `json:"mcpServers,omitempty"`
}

// HookGroup is a matcher with one or more hook definitions.
type HookGroup struct {
	Matcher string    `json:"matcher"`
	Hooks   []HookDef `json:"hooks"`
}

// HookDef defines a single hook command.
type HookDef struct {
	Type    string `json:"type"`
	Command string `json:"command"`
	Timeout int    `json:"timeout,omitempty"`
}

// MCPServer defines an MCP server registration.
type MCPServer struct {
	Type    string            `json:"type,omitempty"`
	Command string            `json:"command"`
	Args    []string          `json:"args"`
	Cwd     string            `json:"cwd,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

func runInstallHooks(cmd *cobra.Command, args []string) error {
	settingsPath := filepath.Join(os.Getenv("HOME"), ".claude", "settings.json")

	settings, err := loadOrCreateSettings(settingsPath)
	if err != nil {
		return err
	}

	if settings.Hooks == nil {
		settings.Hooks = make(map[string][]HookGroup)
	}

	projectRoot := findProjectRoot()

	// PreToolUse hook — router dispatch & policy enforcement
	settings.Hooks["PreToolUse"] = []HookGroup{
		{
			Matcher: "*",
			Hooks: []HookDef{
				{
					Type:    "command",
					Command: fmt.Sprintf("python3 %s/hooks/pre_tool_use.py", projectRoot),
					Timeout: 10,
				},
			},
		},
	}

	// PostToolUse hook — metering logger
	settings.Hooks["PostToolUse"] = []HookGroup{
		{
			Matcher: "*",
			Hooks: []HookDef{
				{
					Type:    "command",
					Command: fmt.Sprintf("python3 %s/hooks/post_tool_use.py", projectRoot),
					Timeout: 10,
				},
			},
		},
	}

	// Stop hook — session summary
	settings.Hooks["Stop"] = []HookGroup{
		{
			Matcher: "",
			Hooks: []HookDef{
				{
					Type:    "command",
					Command: fmt.Sprintf("python3 %s/hooks/stop.py", projectRoot),
					Timeout: 10,
				},
			},
		},
	}

	if err := saveSettings(settingsPath, settings); err != nil {
		return err
	}

	fmt.Println("Hooks installed successfully:")
	fmt.Println("  PreToolUse  → hooks/pre_tool_use.py (router dispatch)")
	fmt.Println("  PostToolUse → hooks/post_tool_use.py (metering logger)")
	fmt.Println("  Stop        → hooks/stop.py (session summary)")
	fmt.Printf("  Settings    → %s\n", settingsPath)
	return nil
}

func runInstallMCP(cmd *cobra.Command, args []string) error {
	settingsPath := filepath.Join(os.Getenv("HOME"), ".claude", "settings.json")

	settings, err := loadOrCreateSettings(settingsPath)
	if err != nil {
		return err
	}

	if settings.McpServers == nil {
		settings.McpServers = make(map[string]MCPServer)
	}

	projectRoot := findProjectRoot()

	settings.McpServers["fabric-worker"] = MCPServer{
		Type:    "stdio",
		Command: "python3",
		Args:    []string{fmt.Sprintf("%s/worker/server.py", projectRoot), "--mcp"},
		Cwd:     projectRoot,
	}

	if err := saveSettings(settingsPath, settings); err != nil {
		return err
	}

	fmt.Println("MCP server registered successfully:")
	fmt.Println("  Name    → fabric-worker")
	fmt.Printf("  Command → python %s/worker/server.py --mcp\n", projectRoot)
	fmt.Printf("  Cwd     → %s\n", projectRoot)
	fmt.Printf("  Settings → %s\n", settingsPath)
	return nil
}

func loadOrCreateSettings(path string) (*ClaudeSettings, error) {
	settings := &ClaudeSettings{}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Create directory if needed
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return nil, fmt.Errorf("failed to create settings directory: %w", err)
			}
			return settings, nil
		}
		return nil, fmt.Errorf("failed to read settings: %w", err)
	}

	if err := json.Unmarshal(data, settings); err != nil {
		return nil, fmt.Errorf("failed to parse settings: %w", err)
	}

	return settings, nil
}

func saveSettings(path string, settings *ClaudeSettings) error {
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal settings: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("failed to write settings: %w", err)
	}

	return nil
}
