package cli

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/spf13/cobra"
	tmuxpkg "github.com/agent-fabric/dispatch/internal/tmux"
)

var sessionCmd = &cobra.Command{
	Use:   "session",
	Short: "Manage tmux sessions with Claude Code + worker panes",
}

var sessionStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start a tmux session with Claude Code and worker visibility panes",
	RunE:  runSessionStart,
}

var sessionStopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop the current fabric tmux session",
	RunE:  runSessionStop,
}

var sessionAttachCmd = &cobra.Command{
	Use:   "attach",
	Short: "Attach to an existing fabric tmux session",
	RunE:  runSessionAttach,
}

var sessionListCmd = &cobra.Command{
	Use:   "list",
	Short: "List active fabric sessions",
	RunE:  runSessionList,
}

func init() {
	sessionCmd.AddCommand(sessionStartCmd)
	sessionCmd.AddCommand(sessionStopCmd)
	sessionCmd.AddCommand(sessionAttachCmd)
	sessionCmd.AddCommand(sessionListCmd)
}

func runSessionStart(cmd *cobra.Command, args []string) error {
	// Check tmux is available
	if _, err := exec.LookPath("tmux"); err != nil {
		return fmt.Errorf("tmux not found in PATH. Install tmux first")
	}

	mgr := tmuxpkg.NewManager()
	return mgr.StartSession()
}

func runSessionStop(cmd *cobra.Command, args []string) error {
	mgr := tmuxpkg.NewManager()
	return mgr.StopSession()
}

func runSessionAttach(cmd *cobra.Command, args []string) error {
	mgr := tmuxpkg.NewManager()
	return mgr.AttachSession()
}

func runSessionList(cmd *cobra.Command, args []string) error {
	listCmd := exec.Command("tmux", "list-sessions", "-F", "#{session_name}: #{session_windows} windows (#{session_created_string})")
	listCmd.Stdout = os.Stdout
	listCmd.Stderr = os.Stderr
	if err := listCmd.Run(); err != nil {
		fmt.Println("No active tmux sessions.")
	}
	return nil
}
