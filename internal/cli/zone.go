package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var zoneCmd = &cobra.Command{
	Use:   "zone",
	Short: "Manage Agent Fabric zones (Phase 2)",
}

var zoneInitCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize a new zone",
	RunE:  runZoneInit,
}

var zoneAddNodeCmd = &cobra.Command{
	Use:   "add-node <address>",
	Short: "Add a node to the current zone",
	Args:  cobra.ExactArgs(1),
	RunE:  runZoneAddNode,
}

var zoneListCmd = &cobra.Command{
	Use:   "list",
	Short: "List nodes in the current zone",
	RunE:  runZoneList,
}

func init() {
	zoneCmd.AddCommand(zoneInitCmd)
	zoneCmd.AddCommand(zoneAddNodeCmd)
	zoneCmd.AddCommand(zoneListCmd)
}

func runZoneInit(cmd *cobra.Command, args []string) error {
	fmt.Println("Zone management is a Phase 2 feature.")
	fmt.Println("Phase 1 runs on a single machine. Complete Phase 1 gates first.")
	return nil
}

func runZoneAddNode(cmd *cobra.Command, args []string) error {
	fmt.Println("Zone management is a Phase 2 feature.")
	return nil
}

func runZoneList(cmd *cobra.Command, args []string) error {
	fmt.Println("Zone management is a Phase 2 feature.")
	return nil
}
