package tmux

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

const (
	SessionName = "fabric"
	MainPane    = "claude"
	WorkerPane  = "worker"
	MeterPane   = "metering"
)

// Manager handles tmux session lifecycle for fabric dispatch.
type Manager struct {
	sessionName string
}

// NewManager creates a new tmux session manager.
func NewManager() *Manager {
	return &Manager{sessionName: SessionName}
}

// StartSession creates a tmux session with Claude Code in the main pane,
// worker logs in a side pane, and metering in a bottom pane.
func (m *Manager) StartSession() error {
	// Check if session already exists
	if m.sessionExists() {
		fmt.Printf("Session '%s' already exists. Attaching...\n", m.sessionName)
		return m.AttachSession()
	}

	meteringLog := filepath.Join(os.Getenv("HOME"), ".fabric", "metering.jsonl")

	// Create the main session with Claude Code
	if err := m.run("new-session", "-d", "-s", m.sessionName, "-n", MainPane); err != nil {
		return fmt.Errorf("failed to create tmux session: %w", err)
	}

	// Send claude command to the main pane
	if err := m.run("send-keys", "-t", fmt.Sprintf("%s:%s", m.sessionName, MainPane), "claude", "Enter"); err != nil {
		return fmt.Errorf("failed to start claude: %w", err)
	}

	// Split right for worker logs (30% width)
	if err := m.run("split-window", "-h", "-t", fmt.Sprintf("%s:%s", m.sessionName, MainPane), "-p", "30"); err != nil {
		return fmt.Errorf("failed to create worker pane: %w", err)
	}

	// Label and start worker log tail in the right pane
	workerTarget := fmt.Sprintf("%s:%s.1", m.sessionName, MainPane)
	if err := m.run("send-keys", "-t", workerTarget,
		fmt.Sprintf("echo '=== Fabric Worker Logs ===' && touch %s && tail -f %s", meteringLog, meteringLog), "Enter"); err != nil {
		return fmt.Errorf("failed to start worker log: %w", err)
	}

	// Split the worker pane horizontally for metering summary
	if err := m.run("split-window", "-v", "-t", workerTarget, "-p", "40"); err != nil {
		return fmt.Errorf("failed to create metering pane: %w", err)
	}

	// Show metering summary in the bottom-right pane
	meterTarget := fmt.Sprintf("%s:%s.2", m.sessionName, MainPane)
	if err := m.run("send-keys", "-t", meterTarget,
		"echo '=== Fabric Metering ===' && watch -n5 'python -m metering.summary 2>/dev/null || echo \"No metering data yet\"'", "Enter"); err != nil {
		return fmt.Errorf("failed to start metering display: %w", err)
	}

	// Select the main Claude Code pane
	if err := m.run("select-pane", "-t", fmt.Sprintf("%s:%s.0", m.sessionName, MainPane)); err != nil {
		return fmt.Errorf("failed to select main pane: %w", err)
	}

	// Set pane titles
	m.run("select-pane", "-t", fmt.Sprintf("%s:%s.0", m.sessionName, MainPane), "-T", "Claude Code")
	m.run("select-pane", "-t", workerTarget, "-T", "Worker Logs")
	m.run("select-pane", "-t", meterTarget, "-T", "Metering")

	// Enable pane border status
	m.run("set-option", "-t", m.sessionName, "pane-border-status", "top")
	m.run("set-option", "-t", m.sessionName, "pane-border-format", " #{pane_title} ")

	fmt.Printf("Fabric session '%s' started.\n", m.sessionName)
	fmt.Println("Layout: Claude Code (left) | Worker Logs (top-right) | Metering (bottom-right)")

	// Attach to the session
	return m.AttachSession()
}

// StopSession kills the fabric tmux session.
func (m *Manager) StopSession() error {
	if !m.sessionExists() {
		fmt.Printf("No active session '%s' found.\n", m.sessionName)
		return nil
	}

	if err := m.run("kill-session", "-t", m.sessionName); err != nil {
		return fmt.Errorf("failed to stop session: %w", err)
	}

	fmt.Printf("Session '%s' stopped.\n", m.sessionName)
	return nil
}

// AttachSession attaches to the fabric tmux session.
func (m *Manager) AttachSession() error {
	if !m.sessionExists() {
		return fmt.Errorf("no active session '%s' found. Run 'fabric session start' first", m.sessionName)
	}

	cmd := exec.Command("tmux", "attach-session", "-t", m.sessionName)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// sessionExists checks if the tmux session exists.
func (m *Manager) sessionExists() bool {
	cmd := exec.Command("tmux", "has-session", "-t", m.sessionName)
	return cmd.Run() == nil
}

// run executes a tmux command with the given arguments.
func (m *Manager) run(args ...string) error {
	cmd := exec.Command("tmux", args...)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
