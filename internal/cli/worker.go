package cli

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
)

var workerCmd = &cobra.Command{
	Use:   "worker",
	Short: "Manage local/remote workers",
}

var workerUpCmd = &cobra.Command{
	Use:   "up",
	Short: "Start a local worker (Ollama-backed MCP server)",
	RunE:  runWorkerUp,
}

var workerDownCmd = &cobra.Command{
	Use:   "down",
	Short: "Stop the running worker",
	RunE:  runWorkerDown,
}

var workerStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show worker status",
	RunE:  runWorkerStatus,
}

var localFlag bool

func init() {
	workerUpCmd.Flags().BoolVar(&localFlag, "local", true, "Start a local worker (default)")
	workerCmd.AddCommand(workerUpCmd)
	workerCmd.AddCommand(workerDownCmd)
	workerCmd.AddCommand(workerStatusCmd)
}

func runWorkerUp(cmd *cobra.Command, args []string) error {
	// Check if ollama is available
	if _, err := exec.LookPath("ollama"); err != nil {
		return fmt.Errorf("ollama CLI not found in PATH. Install from https://ollama.com")
	}

	// Check if ollama is serving
	checkCmd := exec.Command("ollama", "list")
	if err := checkCmd.Run(); err != nil {
		fmt.Println("Starting ollama serve...")
		serveCmd := exec.Command("ollama", "serve")
		serveCmd.Stdout = os.Stdout
		serveCmd.Stderr = os.Stderr
		if err := serveCmd.Start(); err != nil {
			return fmt.Errorf("failed to start ollama: %w", err)
		}
	}

	// Start the Python MCP worker
	fmt.Println("Starting fabric-worker MCP server...")
	pythonCmd := exec.Command("python", "-m", "worker.server")
	pythonCmd.Dir = findProjectRoot()
	pythonCmd.Stdout = os.Stdout
	pythonCmd.Stderr = os.Stderr
	return pythonCmd.Run()
}

func runWorkerDown(cmd *cobra.Command, args []string) error {
	fmt.Println("Stopping fabric-worker...")
	// Send SIGTERM to the worker process
	killCmd := exec.Command("pkill", "-f", "worker.server")
	return killCmd.Run()
}

func runWorkerStatus(cmd *cobra.Command, args []string) error {
	// Check ollama status
	fmt.Println("=== Ollama Status ===")
	ollamaCmd := exec.Command("ollama", "list")
	ollamaCmd.Stdout = os.Stdout
	ollamaCmd.Stderr = os.Stderr
	if err := ollamaCmd.Run(); err != nil {
		fmt.Println("Ollama: not running")
	}

	// Check worker process
	fmt.Println("\n=== Worker Status ===")
	pgrepCmd := exec.Command("pgrep", "-f", "worker.server")
	if err := pgrepCmd.Run(); err != nil {
		fmt.Println("Worker: not running")
	} else {
		fmt.Println("Worker: running")
	}
	return nil
}

func findProjectRoot() string {
	// Walk up from executable to find project root
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	return dir
}
