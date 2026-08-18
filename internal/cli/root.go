package cli

import (
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "fabric",
	Short: "Agent Fabric Dispatch — policy-driven task router for Claude Code",
	Long: `fabric is the CLI for agent-fabric-dispatch.
It sits beside your Claude Code session and dispatches sub-tasks to local/remote
workers based on configurable policy rules, saving tokens while keeping quality high.`,
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.AddCommand(workerCmd)
	rootCmd.AddCommand(installHooksCmd)
	rootCmd.AddCommand(installMCPCmd)
	rootCmd.AddCommand(policyCmd)
	rootCmd.AddCommand(sessionCmd)
	rootCmd.AddCommand(zoneCmd)
	rootCmd.AddCommand(statusCmd)
}
